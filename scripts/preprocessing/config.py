from scripts.utils import get_root_path

ROOT_PATH = get_root_path()


class GroupingRawData:
    DATA_PARENT_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104'
    DATA_RAW_FILE_NAME = 'spikeprot0104.fasta'
    SPLIT_FILES_DIR_NAME = 'split_data'
    FILES_NAME_CORE = 'spikeprot_batch_data'
    DIVISION_TECHNIQUE = 'month'
    PERIOD_ROOT_DIR_NAME = 'periods'


class Clustering:
    SPLIT_FILES_DIR_NAME = 'split_data'
    DATA_PERIODS_PATH = f'{GroupingRawData.DATA_PARENT_PATH}/{GroupingRawData.SPLIT_FILES_DIR_NAME}/periods'

