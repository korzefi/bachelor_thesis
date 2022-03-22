from scripts.utils import get_root_path
from enum import Enum

ROOT_PATH = get_root_path()


class MakingClusters:

    DATA_PARENT_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104'
    EMBEDDED_DATA_PATH = f'{DATA_PARENT_PATH}/split_data/periods/unique/sequences_as_vectors'
    CLUSTER_PLOT_PATH = f'{DATA_PARENT_PATH}/clusters_plots'
    CLUSTER_METHOD = 'KMeans'
    REDUCTION_METHOD = 'PCA'
    PLOT_CLUSTERS_DIMS = 2
    INIT_N_CLUSTERS = 2
    END_N_CLUSTERS = 30

class KMeansConfig:
    """N_ITERATIONS_INIT - number of iterations with random initializations of centroids,
        where the one with the smallest distortion (cost function) is picked"""
    N_ITERATIONS_INIT = 10000

