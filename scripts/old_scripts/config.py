# author: Filip Korzeniewski


from scripts.utils import get_root_path

ROOT_PATH = get_root_path()


DATA_PARENT_PATH = ROOT_PATH + '/data/input'


class GroupingRawData:
    DATA_RAW_FILE_NAME = 'spikeprot_shortened.fasta'
    # DIVISION might be: month, quarter, year
    DIVISION_TECHNIQUE = 'month'
    PERIOD_UNIQUE_DIR_NAME = 'unique'


class Clustering:
    PROT_VEC_PATH = ROOT_PATH + f'{DATA_PARENT_PATH}/protVec_100d_3grams.csv'
    DATA_PERIODS_UNIQUE_PATH = f'{DATA_PARENT_PATH}/split_data/periods/unique'


class CreatingDatasets:
    """epitopes - list of positions: [(<beginning epitope position>, <end epitope position>)]
       dataset_size - number of rows in final dataset
       mutated_data_ratio - dataset is being created as long as
                               the mutated number of data rows in total number of rows is fulfilled
       window_size - number of consecutive clusters to be linked
       sample_num_per_pos - number of samples to be generated per epitope position
       epitopes_similarity_threshold - threshold over which sequences are replaced - check readme"""
    EPITOPES = [(194, 210), (291, 325), (307, 323), (371, 387), (410, 426), (525, 566), (530, 544), (722, 739),
                (747, 763), (749, 771), (754, 770), (891, 906), (897, 913), (1101, 1115), (1129, 1145), (1213, 1229)]
    MUTATED_DATA_RATIO = 0.2
    WINDOW_SIZE = 10
    # 317 positions -> samples_num * 317 positions -> final number of rows
    # DATASET_SIZE = 634000
    DATASET_SIZE = 317 * 40
    EPITOPES_SIMILARITY_THRESHOLD = 0.5
    DATASETS_MAIN_FILE_PATH = f'{DATA_PARENT_PATH}/datasets/period-{GroupingRawData.DIVISION_TECHNIQUE}.csv'
    DATASETS_DIR_PATH = f'{DATA_PARENT_PATH}/datasets'


class ClusterToProceed:
    FILES_CLUSTERS_NUM = {'2020-2.csv': 2, '2020-3.csv': 2, '2020-4.csv': 2,
                          '2020-5.csv': 2, '2020-6.csv': 2, '2020-7.csv': 3, '2020-8.csv': 3, '2020-9.csv': 4,
                          '2020-10.csv': 4, '2020-11.csv': 5, '2020-12.csv': 6, '2021-1.csv': 7, '2021-2.csv': 8,
                          '2021-3.csv': 8, '2021-4.csv': 8, '2021-5.csv': 9, '2021-6.csv': 10, '2021-7.csv': 11,
                          '2021-8.csv': 11, '2021-9.csv': 12, '2021-10.csv': 14, '2021-11.csv': 15,
                          '2021-12.csv': 16, '2022-1.csv': 18}
