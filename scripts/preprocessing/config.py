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


