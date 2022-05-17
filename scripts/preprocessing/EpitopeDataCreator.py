import logging
import multiprocessing
import os

import pandas as pd
from sklearn.utils import shuffle
import random

from natsort import natsorted

from scripts.preprocessing.config import CreatingDatasets, Clustering, GroupingRawData


class SequencesBatchToMutated(Exception):
    def __init__(self, message="Could not find sequences fulfilling threshold criterion"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class EpitopeDataCreator:
    CONTEXT_SIZE = 2

    def __init__(self,
                 window_size=CreatingDatasets.WINDOW_SIZE,
                 threshold=CreatingDatasets.EPITOPES_SIMILARITY_THRESHOLD,
                 dataset_size=CreatingDatasets.DATASET_SIZE):
        self.dataset_size = dataset_size
        self.__window_size = window_size
        self.__epitopes_positions = self.__parse_epitope_positions(CreatingDatasets.EPITOPES)
        self.__max_similar_epitopes = int(len(self.__epitopes_positions) * threshold)

    def __parse_epitope_positions(self, epitope_positions):
        epitopes = [list(range(start, end + 1)) for start, end in epitope_positions]
        flat_epitopes = [positions for epitope in epitopes for positions in epitope]
        # transform positions to list positions (starting from 0)
        flat_epitopes = list(map(lambda x: x - 1, flat_epitopes))
        return flat_epitopes

    def set_dataset_size(self, dataset_size):
        self.dataset_size = dataset_size

    def create_data(self):
        df = pd.read_csv(Clustering.CLUSTERS_CENTROIDS_DATA_PATH)
        windows = self.__get_windows(df['period'])
        samples = self.__create_samples(df, windows)
        df = self.__create_final_dataset(samples)
        return df

    def __get_windows(self, periods: pd.Series) -> [{}]:
        periods_num = periods.nunique()
        unique_periods = periods.unique()
        sorted_unique_periods = natsorted(unique_periods)
        # 1 due to the y column
        dataset_row_len = self.__window_size + 1
        if periods_num < dataset_row_len:
            logging.warning('Periods number is lower than given window size')
            logging.warning(f'Window size is set for periods_num = {periods_num}')
            self.__window_size = periods_num - 1
        windows_num = periods_num - self.__window_size
        result = []
        for i in range(windows_num):
            result.append({'x': sorted_unique_periods[i:i + self.__window_size]})
            result[i]['y'] = sorted_unique_periods[i + self.__window_size]
        return result

    def __create_samples(self, centroids_df, windows):
        epitopes_pos_num = 317
        samples_num_per_pos = self.dataset_size // epitopes_pos_num
        num_samples_per_window = samples_num_per_pos // len(windows)
        reminder = samples_num_per_pos - len(windows) * num_samples_per_window
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
        result = []
        mutated_to_much = True
        while mutated_to_much:
            try:
                first_seq_dict = self.__choose_first(centroids_df, window)
                result = self.__link_rest_clusters(centroids_df, first_seq_dict, window)
                mutated_to_much = False
            except SequencesBatchToMutated:
                mutated_to_much = True
        return result

    def __choose_first(self, centroids_df, window: {}):
        first_period_str = window['x'][0]
        first_period_row = centroids_df[centroids_df['period'] == first_period_str].sample()
        current_cluster = first_period_row.iloc[0]['cluster']
        next_clusters = first_period_row.iloc[0]['next_cluster'].split('-')
        seq = self.__get_sequence(f'{first_period_str}.csv', current_cluster)
        return {'seq': seq, 'next_clusters': next_clusters}

    def __link_rest_clusters(self, centroids_df, first_seq_dict, window):
        result = [first_seq_dict['seq']]
        next_clusters = first_seq_dict['next_clusters']
        current_cluster = random.choice(next_clusters)
        for i in range(1, len(window['x'])):
            current_period_str = window['x'][i]
            next_clusters_row = centroids_df[(centroids_df['period'] == current_period_str) &
                                             (centroids_df['cluster'] == int(current_cluster))]
            next_clusters = next_clusters_row.iloc[0]['next_cluster'].split('-')
            seq = self.__get_sequence(f'{current_period_str}.csv', int(current_cluster))
            prev_seq = result[-1]
            max_tries_num_count = 0
            while self.__is_mutated_to_much(prev_seq, seq):
                max_tries_num_count += 1
                seq = self.__get_sequence(f'{current_period_str}.csv', int(current_cluster))
                if max_tries_num_count > 10:
                    raise SequencesBatchToMutated
            result.append(seq)
            current_cluster = random.choice(next_clusters)
        current_period_str = window['y']
        seq = self.__get_sequence(f'{current_period_str}.csv', int(current_cluster))
        result.append(seq)
        return result

    def __get_sequence(self, filename, current_cluster):
        expected_seq_len_with_asterisk = 1274
        filepath = f'{Clustering.DATA_PERIODS_UNIQUE_PATH}/{filename}'
        df = pd.read_csv(filepath)
        current_cluster_rows = df[(df['cluster'] == current_cluster) &
                                  (df['sequence'].str.len() == expected_seq_len_with_asterisk)]
        if current_cluster_rows.empty:
            current_cluster_rows = df[df['cluster'] == current_cluster]
        chosen_row = current_cluster_rows.sample()
        seq = chosen_row.iloc[0]['sequence']
        return seq

    def __is_mutated_to_much(self, prev_seq, seq):
        mutated_epitopes = 0
        for pos in self.__epitopes_positions:
            if prev_seq[pos] != seq[pos]:
                mutated_epitopes += 1
        return mutated_epitopes > self.__max_similar_epitopes

    def __create_final_dataset(self, samples_seqs):
        logging.info('Transferring samples into dataset')
        cut_out_epitopes_samples = self.__cut_out_epitopes_with_context(samples_seqs)
        transformed_epitopes = self.__transform_to_protvec_positions(cut_out_epitopes_samples)
        df = self.__transform_to_datasets(transformed_epitopes)
        return df

    def __cut_out_epitopes_with_context(self, sample_seqs) -> [[[]]]:
        cxt_size = EpitopeDataCreator.CONTEXT_SIZE
        cut_out_epitopes_samples = []
        for sample in sample_seqs:
            new_sample = []
            for seq in sample:
                seqs_epitopes = []
                for position in self.__epitopes_positions:
                    seqs_epitopes.append(seq[position - cxt_size:position + cxt_size + 1])
                new_sample.append(seqs_epitopes)
            cut_out_epitopes_samples.append(new_sample)
        return cut_out_epitopes_samples

    def __transform_to_protvec_positions(self, epitopes_samples: [[[]]]) -> [[[[]]]]:
        epitopes_pos_num = 317
        samples_num_per_pos = self.dataset_size // epitopes_pos_num
        protvec = pd.read_csv(Clustering.PROT_VEC_PATH)
        index_samples = []
        for count, sample in enumerate(epitopes_samples):
            logging.info(f'Transfering {count + 1} seq out of {samples_num_per_pos}')
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
        logging.info('Transforming samples to datasets')
        df = self.__create_dataset_dataframe()
        epitopes_samples_num = len(epitopes_samples)
        for counter, sample in enumerate(epitopes_samples):
            logging.info(f'Processing {counter + 1} sample out of {epitopes_samples_num}')
            seq_len = len(sample[0])
            for i in range(seq_len):
                row = []
                for seq in sample:
                    row.append(seq[i])
                row = self.__adjust_row_for_dataset(row)
                df.loc[len(df) + 1] = row
        filepath = CreatingDatasets.DATASETS_MAIN_FILE_PATH
        logging.info('Dataset created, shuffling data...')
        df = shuffle(df)
        df.reset_index(drop=True, inplace=True)
        return df

    def __create_dataset_dataframe(self):
        columns = ['y']
        columns_x = [str(i) for i in range(self.__window_size)]
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


class DatasetRefiller:
    MAX_ITERS = 10
    PROCESSES_NUM = 20

    def __init__(self,
                 epitope_creator: EpitopeDataCreator,
                 duplicated_data_ratio=CreatingDatasets.DUPLICATED_DATA_RATIO):
        self.__creator = epitope_creator
        self.__duplicated_data_ratio = duplicated_data_ratio
        self.duplicates_subset = [str(i) for i in range(CreatingDatasets.WINDOW_SIZE)]

    def create_dataset(self):
        logging.info('Creating final dataset using refiller')
        filepath = CreatingDatasets.DATASETS_MAIN_FILE_PATH
        df = self.__creator.create_data()
        df.reset_index(drop=True, inplace=True)
        df.to_csv(filepath, index=False)
        logging.info(f'1 of {DatasetRefiller.MAX_ITERS} iteration complete')
        for i in range(DatasetRefiller.MAX_ITERS - 1):
            df.drop_duplicates(subset=self.duplicates_subset, inplace=True)
            df.reset_index(drop=True, inplace=True)
            new_dataset_size = CreatingDatasets.DATASET_SIZE - len(df)
            self.__creator.set_dataset_size(new_dataset_size)
            new_batch = self.__creator.create_data()
            df = pd.concat([df, new_batch], ignore_index=True)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(filepath, index=False)
            total_dataset_size = CreatingDatasets.DATASET_SIZE
            df_no_duplicates = df.drop_duplicates(subset=self.duplicates_subset, inplace=False)
            actual_ratio = (len(df_no_duplicates) * 100) / total_dataset_size
            logging.info(f'{i+2} iteration of {DatasetRefiller.MAX_ITERS} iteration complete')
            logging.info(f'Duplicated ratio: {actual_ratio}')
            if actual_ratio > self.__duplicated_data_ratio:
                logging.info('Duplicated ratio achieved')
                return

    def create_datasets_multiprocess(self):
        processes = []
        for i in range(DatasetRefiller.PROCESSES_NUM):
            p = multiprocessing.Process(target=self._create_dataset_file, args=(i,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.__merge_dataset_files()

    def merge_files(self):
        self.__merge_dataset_files()

    def _create_dataset_file(self, process_num):
        logging.basicConfig(level=logging.INFO)

        filepath = CreatingDatasets.DATASETS_MAIN_FILE_PATH
        filepath = filepath[:-4] + f'-{process_num}.csv'
        df = self.__creator.create_data()
        df.to_csv(filepath, index=False)
        logging.info(f'Creating file for {process_num} process finished')

    def __merge_dataset_files(self):
        filepath = CreatingDatasets.DATASETS_DIR_PATH
        periods = os.listdir(filepath)
        periods = list(map(lambda x: f'{CreatingDatasets.DATASETS_DIR_PATH}/{x}', periods))
        df = pd.read_csv(periods[0])
        df.drop_duplicates(subset=self.duplicates_subset, inplace=True)
        df.reset_index(drop=True, inplace=True)
        periods = periods[1:]
        for filename in periods:
            current_df = pd.read_csv(filename)
            df = pd.concat([df, current_df], ignore_index=True)
            df.drop_duplicates(subset=self.duplicates_subset, inplace=True)
            df.reset_index(drop=True, inplace=True)
        output_filepath = f'{CreatingDatasets.DATASETS_DIR_PATH}/period-month-concat.csv'
        df.to_csv(output_filepath, index=False)


def create_final_data():
    epitope_creator = EpitopeDataCreator()
    dataset_creator = DatasetRefiller(epitope_creator=epitope_creator)
    # dataset_creator.create_dataset()
    # dataset_creator.create_datasets_multiprocess()
    dataset_creator.merge_files()
