from scripts.utils import get_root_path

ROOT_PATH = get_root_path()


class GroupingRawData:
    DATA_PARENT_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104'
    DATA_RAW_FILE_NAME = 'spikeprot0104.fasta'
    SPLIT_FILES_DIR_NAME = 'split_data'
    FILES_NAME_CORE = 'spikeprot_batch_data'
    DIVISION_TECHNIQUE = 'month'
    PERIOD_ROOT_DIR_NAME = 'periods'
    PERIOD_UNIQUE_DIR_NAME = 'unique'


class Clustering:
    DATA_PERIODS_UNIQUE_PATH = f'{GroupingRawData.DATA_PARENT_PATH}/' \
                               f'{GroupingRawData.SPLIT_FILES_DIR_NAME}/' \
                               f'{GroupingRawData.PERIOD_ROOT_DIR_NAME}/' \
                               f'{GroupingRawData.PERIOD_UNIQUE_DIR_NAME}'
    PROT_VEC_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104/protVec_100d_3grams.csv'
    VECTOR_TEMP_DIR_NAME = 'sequences_as_vectors'
    VECTOR_TEMP_DIR_PATH = f'{DATA_PERIODS_UNIQUE_PATH}/{VECTOR_TEMP_DIR_NAME}'
    CLUSTERS_CENTROIDS_DATA_PATH = f'{DATA_PERIODS_UNIQUE_PATH}/clusters_centroids_data.csv'


class CreatingDatasets:
    """epitopes - list of positions: [(<beginning epitope position>, <end epitope position>)]
       window_size - number of consecutive clusters to be linked
       sample_num_per_pos - number of samples to be generated per epitope position"""
    EPITOPES = [(194, 210), (291, 325), (307, 323), (371, 387), (410, 426), (525, 566), (530, 544), (722, 739),
                (747, 763), (749, 771), (754, 770), (891, 906), (897, 913), (1101, 1115), (1129, 1145), (1213, 1229)]
    WINDOW_SIZE = 5
    # 317 positions -> samples_num * 317 positions -> final number of rows
    # SAMPLES_NUM_PER_POS = 1000
    SAMPLES_NUM_PER_POS = 10
    # TODO: later can be done, for now just randomly
    # STRATEGY = 'single_epitope'
    # ANALYZED_EPITOPE = 194


class ClusterToProceed:
    FILES_CLUSTERS_NUM = {'2019-12.csv': 3, '2020-1.csv': 7, '2020-2.csv': 3, '2020-3.csv': 2, '2020-4.csv': 2,
                          '2020-5.csv': 2, '2020-6.csv': 2, '2020-7.csv': 7, '2020-8.csv': 7, '2020-9.csv': 7,
                          '2020-10.csv': 7, '2020-11.csv': 7, '2020-12.csv': 7, '2021-1.csv': 7, '2021-2.csv': 7,
                          '2021-3.csv': 7, '2021-4.csv': 7, '2021-5.csv': 7, '2021-6.csv': 7, '2021-7.csv': 7,
                          '2021-8.csv': 7, '2021-9.csv': 7, '2021-10.csv': 7, '2021-11.csv': 7, '2021-12.csv': 7}


