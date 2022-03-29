import logging

from scripts.clustering.config import MakingClusters as cfg
from scripts.clustering.ClusterCreator import ClusterCreator, ClusterMethodFactory
from scripts.clustering.visualization import ReductionMethodFactory, plot_clusters, plot_elbow_method

FILENAME_TO_BE_PROCEED = '2020-3.csv'
USE_RANGE_CLUSTERS = False
# this will be skipped if USE_RANGE_CLUSTERS is set, otherwise there will be only N_CLUSTERS version created
N_CLUSTERS = 2
SAVE_FIG = False


class ClusterCreatorFactory:
    @staticmethod
    def create():
        method_factory = ClusterMethodFactory()
        cluster_method = method_factory.create_method(cfg.CLUSTER_METHOD)
        return ClusterCreator(cluster_method)


def create_clusters():
    cluster_creator = ClusterCreatorFactory.create()
    filepath = f'{cfg.EMBEDDED_DATA_PATH}/{FILENAME_TO_BE_PROCEED}'
    return cluster_creator.create_clusters(filepath=filepath, use_range=USE_RANGE_CLUSTERS, n_clusters=N_CLUSTERS)


def visualize():
    reduction_factory = ReductionMethodFactory()
    reduction_method = reduction_factory.create_method(cfg.REDUCTION_METHOD)
    clusters = create_clusters()
    logging.info(f'Clusters created')
    if USE_RANGE_CLUSTERS is True:
        plot_elbow_method(clusters)
    else:
        filename = f'{FILENAME_TO_BE_PROCEED[:-4]}-c{N_CLUSTERS}'
        plot_clusters(clusters, reduction_method, filename, save_fig=SAVE_FIG)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Clustering method: {cfg.CLUSTER_METHOD}')
    logging.info(f'Reduction method: {cfg.REDUCTION_METHOD}')
    visualize()
    logging.info("Done")
