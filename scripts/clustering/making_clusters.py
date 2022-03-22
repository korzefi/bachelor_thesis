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
    filename = '2019-12.csv'
    cluster_creator = ClusterCreatorFactory.create()
    return cluster_creator.create_clusters(filename)


def visualize():
    reduction_factory = ReductionMethodFactory()
    reduction_method = reduction_factory.create_method(cfg.REDUCTION_METHOD)
    clusters = create_clusters()
    plot_clusters(clusters, reduction_method)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    visualize()
    logging.info("Done")
