import logging
import pandas as pd

from natsort import natsorted

from scripts.preprocessing.config import CreatingDatasets, Clustering


class EpitopeDataCreator:
    def __init__(self, window_size=CreatingDatasets.WINDOW_SIZE):
        self.window_size = window_size
        self.epitopes_positions = self.__parse_epitope_positions(CreatingDatasets.EPITOPES)

    def create_data(self):
        df = pd.read_csv(Clustering.CLUSTERS_CENTROIDS_DATA_PATH)
        windows = self.__get_windows(df['period'])
        self.__create_samples(df, windows)

    def __parse_epitope_positions(self, epitope_positions):
        epitopes = [list(range(start, end + 1)) for start, end in epitope_positions]
        flat_epitopes = [positions for epitope in epitopes for positions in epitope]
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

    def __create_samples(self, df, windows):
        num_samples_per_window = CreatingDatasets.SAMPLES_NUM_PER_POS // len(windows)
        reminder = CreatingDatasets.SAMPLES_NUM_PER_POS - len(windows) * num_samples_per_window
        # 1st need to be chosen randomly (random cluster),
        # next from x, and y need to be taken accordingly to pattern (link)
        for window in windows:
            for i in range(num_samples_per_window):
                first_period_str = window['x'][0]
                first_period_row = df[df['period'] == first_period_str].sample()
                current_cluster = first_period_row['cluster']
                next_clusters = first_period_row['next_cluster'].split('-')
                seq = self.__get_sequence(f'{first_period_str}.csv', current_cluster)

    def __get_sequence(self, filename, current_cluster):
        filepath = f'{Clustering.DATA_PERIODS_UNIQUE_PATH}/{filename}'
        df = pd.read_csv(filepath)
        current_cluster_rows = df[df['cluster'] == current_cluster]
        return current_cluster_rows['sequence'].sample()

    def __link_clusters(self):
        pass


def create_final_data():
    creator = EpitopeDataCreator()
    creator.create_data()
