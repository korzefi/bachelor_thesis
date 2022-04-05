from scripts.utils import get_root_path

ROOT_PATH = get_root_path()


class MakingClusters:
    REDUCTION_METHOD = 'PCA'
    CLUSTER_METHOD = 'KMeans'
    PLOT_CLUSTERS_DIMS = 2
    INIT_N_CLUSTERS = 2
    END_N_CLUSTERS = 20
    DATA_PARENT_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104'
    EMBEDDED_DATA_PATH = f'{DATA_PARENT_PATH}/split_data/periods/unique/sequences_as_vectors'
    CLUSTER_PLOT_PATH = f'{DATA_PARENT_PATH}/clusters_plots/{REDUCTION_METHOD}'


class KMeansConfig:
    """N_ITERATIONS_INIT - number of iterations with random initializations of centroids,
        where the one with the smallest distortion (cost function) is picked"""
    N_ITERATIONS_INIT = 3000
    MAX_ITER = 500


class TsneConfig:
    N_ITER = 1000
    N_ITER_WITHOUT_PROGRESS = 300


class LinkingClusters:
    """epitopes - list of positions: [(<beginning epitope position>, <end epitope position>)]
       window_size - number of consecutive clusters to be linked
       sample_num_per_pos - number of samples to be generated per epitope position"""
    EPITOPES = [(194, 210), (291, 325), (307, 323), (371, 387), (410, 426), (525, 566), (530, 544), (722, 739),
                (747, 763), (749, 771), (754, 770), (891, 906), (897, 913), (1101, 1115), (1129, 1145), (1213, 1229)]
    WINDOW_SIZE = 10
    # 317 positions -> samples_num * 317 positions -> final number of rows
    SAMPLES_NUM_PER_POS = 1000
    STRATEGY = 'single_epitope'
    ANALYZED_EPITOPE = 194


class ClusterToProceed:
    FILES_CLUSTERS_NUM = {'2019-12.csv': 3, '2020-1.csv': 7, '2020-2.csv': 3, '2020-3.csv': 2, '2020-4.csv': 2,
                          '2020-5.csv': 2, '2020-6.csv': 2, '2020-7.csv': 7, '2020-8.csv': 7, '2020-9.csv': 7,
                          '2020-10.csv': 7, '2020-11.csv': 7, '2020-12.csv': 7, '2021-1.csv': 7, '2021-2.csv': 7,
                          '2021-3.csv': 7, '2021-4.csv': 7, '2021-5.csv': 7, '2021-6.csv': 7, '2021-7.csv': 7,
                          '2021-8.csv': 7, '2021-9.csv': 7, '2021-10.csv': 7, '2021-11.csv': 7, '2021-12.csv': 7}
