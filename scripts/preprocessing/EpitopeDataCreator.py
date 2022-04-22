import logging

import pandas as pd
from sklearn.utils import shuffle
import random

from natsort import natsorted

from scripts.preprocessing.config import CreatingDatasets, Clustering


class EpitopeDataCreator:
    CONTEXT_SIZE = 2
    def __init__(self, window_size=CreatingDatasets.WINDOW_SIZE):
        self.window_size = window_size
        self.epitopes_positions = self.__parse_epitope_positions(CreatingDatasets.EPITOPES)

    def create_data(self):
        df = pd.read_csv(Clustering.CLUSTERS_CENTROIDS_DATA_PATH)
        windows = self.__get_windows(df['period'])
        samples = self.__create_samples(df, windows)
        self.__create_final_dataset(samples)

    def __parse_epitope_positions(self, epitope_positions):
        epitopes = [list(range(start, end + 1)) for start, end in epitope_positions]
        flat_epitopes = [positions for epitope in epitopes for positions in epitope]
        # transform positions to list positions (starting from 0)
        flat_epitopes = list(map(lambda x: x-1, flat_epitopes))
        return flat_epitopes

    def __get_windows(self, periods: pd.Series) -> [{}]:
        periods_num = periods.nunique()
        unique_periods = periods.unique()
        sorted_unique_periods = natsorted(unique_periods)
        # 1 due to the y column
        dataset_row_len = self.window_size + 1
        if periods_num < dataset_row_len:
            logging.warning('Periods number is lower than given window size')
            logging.warning(f'Window size is set for periods_num = {periods_num}')
            self.window_size = periods_num - 1
        windows_num = periods_num - self.window_size
        result = []
        for i in range(windows_num):
            result.append({'x': sorted_unique_periods[i:i + self.window_size]})
            result[i]['y'] = sorted_unique_periods[i + self.window_size]

        return result

    def __create_samples(self, centroids_df, windows):
        num_samples_per_window = CreatingDatasets.SAMPLES_NUM_PER_POS // len(windows)
        reminder = CreatingDatasets.SAMPLES_NUM_PER_POS - len(windows) * num_samples_per_window
        # 1st need to be chosen randomly (random cluster),
        # next from x, and y need to be taken accordingly to pattern (link)
        sequences = []
        for window in windows:
            logging.info(f'Creating samples for window {window}')
            for i in range(num_samples_per_window):
                sequences.append(self.__link_clusters(centroids_df, window))
        for i in range(reminder):
            sequences.append(self.__link_clusters(centroids_df, windows[-1]))
        return sequences

    def __link_clusters(self, centroids_df, window: {}):
        first_seq_dict = self.__choose_first(centroids_df, window)
        next_clusters = first_seq_dict['next_clusters']
        chosen_cluster = random.choice(next_clusters)
        rest_seqs = self.__link_rest_clusters(centroids_df, chosen_cluster, window)
        return [first_seq_dict['seq']] + rest_seqs

    def __choose_first(self, centroids_df, window: {}):
        first_period_str = window['x'][0]
        first_period_row = centroids_df[centroids_df['period'] == first_period_str].sample()
        current_cluster = first_period_row.iloc[0]['cluster']
        next_clusters = first_period_row.iloc[0]['next_cluster'].split('-')
        seq = self.__get_sequence(f'{first_period_str}.csv', current_cluster)
        return {'seq': seq, 'next_clusters': next_clusters}

    def __link_rest_clusters(self, centroids_df, chosen_cluster, window):
        result = []
        current_cluster = chosen_cluster
        for i in range(1, len(window['x'])):
            current_period_str = window['x'][i]
            next_clusters_row = centroids_df[(centroids_df['period'] == current_period_str) &
                                             (centroids_df['cluster'] == int(current_cluster))]
            next_clusters = next_clusters_row.iloc[0]['next_cluster'].split('-')
            seq = self.__get_sequence(f'{current_period_str}.csv', int(current_cluster))
            result.append(seq)
            current_cluster = random.choice(next_clusters)
        current_period_str = window['y']
        seq = self.__get_sequence(f'{current_period_str}.csv', int(current_cluster))
        result.append(seq)
        return result

    def __get_sequence(self, filename, current_cluster):
        filepath = f'{Clustering.DATA_PERIODS_UNIQUE_PATH}/{filename}'
        df = pd.read_csv(filepath)
        current_cluster_rows = df[df['cluster'] == current_cluster]
        chosen_row = current_cluster_rows.sample()
        seq = chosen_row.iloc[0]['sequence']
        return seq

    def __create_final_dataset(self, samples_seqs):
        logging.info('Transferring samples into dataset')
        cut_out_epitopes_samples = self.__cut_out_epitopes_with_context(samples_seqs)
        transformed_epitopes = self.__transform_to_protvec_positions(cut_out_epitopes_samples)
        self.__transform_to_datasets(transformed_epitopes)

    def __cut_out_epitopes_with_context(self, sample_seqs) -> [[[]]]:
        cxt_size = EpitopeDataCreator.CONTEXT_SIZE
        cut_out_epitopes_samples = []
        for sample in sample_seqs:
            new_sample = []
            for seq in sample:
                seqs_epitopes = []
                for position in self.epitopes_positions:
                    seqs_epitopes.append(seq[position-cxt_size:position+cxt_size+1])
                new_sample.append(seqs_epitopes)
            cut_out_epitopes_samples.append(new_sample)
        return cut_out_epitopes_samples

    def __transform_to_protvec_positions(self, epitopes_samples: [[[]]]) -> [[[[]]]]:
        protvec = pd.read_csv(Clustering.PROT_VEC_PATH)
        index_samples = []
        for count, sample in enumerate(epitopes_samples):
            logging.info(f'Transfering {count+1} seq out of {CreatingDatasets.SAMPLES_NUM_PER_POS}')
            new_sample = []
            for seq in sample:
                seqs_triplets = []
                for epitope_cxt in seq:
                    epitope_triplet_list = self.__epitope_cxt_to_triplet_list(epitope_cxt, protvec)
                    seqs_triplets.append(epitope_triplet_list)
                new_sample.append(seqs_triplets)
            index_samples.append(new_sample)
        return index_samples

    def __epitope_cxt_to_triplet_list(self, epitope_cxt, protvec):
        sites_per_pos_num = 1 + (2 * EpitopeDataCreator.CONTEXT_SIZE)
        triplets_num = sites_per_pos_num - 2
        triplets = [epitope_cxt[i:i + 3] for i in range(triplets_num)]
        return [protvec.index[protvec['words'] == triplet].tolist()[0] for triplet in triplets]

    def __transform_to_datasets(self, epitopes_samples):
        df = self.__create_dataset_dataframe()
        for sample in epitopes_samples:
            seq_len = len(sample[0])
            for i in range(seq_len):
                row = []
                for seq in sample:
                    row.append(seq[i])
                row = self.__adjust_row_for_dataset(row)
                df.loc[len(df) + 1] = row
        filepath = CreatingDatasets.DATASETS_DIR_PATH
        df = shuffle(df)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(filepath, index=False, header=True)

    def __create_dataset_dataframe(self):
        columns = ['y']
        columns_x = [str(i) for i in range(self.window_size)]
        columns += columns_x
        df = pd.DataFrame(columns=columns)
        df.reset_index(drop=True, inplace=True)
        return df

    def __adjust_row_for_dataset(self, row):
        result = row[:-1]
        y_val = row[-1]
        last_period = row[-2]
        val = int
        if len(set(y_val) & set(last_period)) == 3:
            val = 0
        else:
            val = 1
        result.insert(0, val)
        return result


def create_final_data():
    creator = EpitopeDataCreator()
    creator.create_data()
