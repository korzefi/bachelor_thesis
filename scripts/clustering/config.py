from scripts.utils import get_root_path

ROOT_PATH = get_root_path()


class MakingClusters:
    REDUCTION_METHOD = 'PCA'
    CLUSTER_METHOD = 'KMeans'
    PLOT_CLUSTERS_DIMS = 2
    INIT_N_CLUSTERS = 10
    END_N_CLUSTERS = 22
    DATA_PARENT_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104'
    EMBEDDED_DATA_PATH = f'{DATA_PARENT_PATH}/split_data/periods/unique/sequences_as_vectors'
    CLUSTER_PLOT_PATH = f'{DATA_PARENT_PATH}/clusters_plots/{REDUCTION_METHOD}'


class KMeansConfig:
    """N_ITERATIONS_INIT - number of iterations with random initializations of centroids,
        where the one with the smallest distortion (cost function) is picked"""
    N_ITERATIONS_INIT = 1000
    MAX_ITER = 500


class TsneConfig:
    N_ITER = 1000
    N_ITER_WITHOUT_PROGRESS = 300
