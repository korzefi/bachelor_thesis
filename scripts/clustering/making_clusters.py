import logging
from config import MakingClusters as cfg
from ClusterCreator import ClusterCreator, ClusterMethodFactory
from visualization import ReductionMethodFactory, plot_clusters


class ClusterCreatorFactory:
    @staticmethod
    def create():
        method_factory = ClusterMethodFactory()
        cluster_method = method_factory.create_method(cfg.CLUSTER_METHOD)
        return ClusterCreator(cluster_method)


def create_clusters():
    # TODO change to many files
    filename = '2020-11.csv'
    cluster_creator = ClusterCreatorFactory.create()
    return cluster_creator.create_clusters(filename)


def visualize():
    reduction_factory = ReductionMethodFactory()
    reduction_method = reduction_factory.create_method(cfg.REDUCTION_METHOD)
    clusters = create_clusters()
    logging.info('Clusters created')
    plot_clusters(clusters, reduction_method)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Clustering method: {cfg.CLUSTER_METHOD}')
    logging.info(f'Reduction method: {cfg.REDUCTION_METHOD}')
    visualize()
    logging.info("Done")
