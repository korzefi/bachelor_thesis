# author: Filip Korzeniewski


from scripts.utils import get_root_path

ROOT_PATH = get_root_path()


class LoadingDatasetsConfig:
    PROT_VEC_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104/protVec_100d_3grams.csv'

    DATASETS_DIRPATH = ROOT_PATH + "/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104/datasets"
    TRAIN_DATASET_PATH = f'{DATASETS_DIRPATH}/train-15-09-22.csv'
    TEST_DATASET_PATH = f'{DATASETS_DIRPATH}/test-15-09-22.csv'


class ResultsConfig:
    RESULTS_DIRPATH = '/Users/filip/Desktop/praca-inz-eiti/covid-rnn/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104/results'
