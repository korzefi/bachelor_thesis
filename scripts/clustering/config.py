from scripts.utils import get_root_path
from enum import Enum

ROOT_PATH = get_root_path()


class MakingClusters:
    DATA_PARENT_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104'
    EMBEDDED_DATA_PATH = f'{DATA_PARENT_PATH}/split_data/periods/unique/sequences_as_vectors'
    CLUSTER_PLOT_PATH = f'{DATA_PARENT_PATH}/clusters_plots'
    CLUSTER_METHOD = 'KMeans'
    REDUCTION_METHOD = 'TSNE'
    PLOT_CLUSTERS_DIMS = 2


class KMeansConfig:
    INITIAL_N_CLUSTERS = 3
