# author: Filip Korzeniewski


from scripts.utils import get_root_path

ROOT_PATH = get_root_path()


class LoadingDatasetsConfig:
    PROT_VEC_PATH = ROOT_PATH + '/data/spikeprot0308/protVec_100d_3grams.csv'

    DATASETS_DIRPATH = ROOT_PATH + "/data/spikeprot0308/datasets"
    TRAIN_DATASET_PATH = f'{DATASETS_DIRPATH}/train-16-03-2024_12-11.csv'
    VALID_DATASET_PATH = f'{DATASETS_DIRPATH}/valid-16-03-2024_12-11.csv'


class ResultsConfig:
    RESULTS_DIRPATH = '/Users/filip/Desktop/praca-inz-eiti/covid-rnn/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104/results'
