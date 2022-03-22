import logging
from config import MakingClusters as cfg
from ClusterCreator import ClusterCreator, ClusterMethodFactory
from visualization import ReductionMethodFactory, plot_clusters, plot_elbow_method


class ClusterCreatorFactory:
    @staticmethod
    def create():
        method_factory = ClusterMethodFactory()
        cluster_method = method_factory.create_method(cfg.CLUSTER_METHOD)
        return ClusterCreator(cluster_method)


def create_clusters():
    # TODO change to many files
    filename = '2020-1.csv'
    cluster_creator = ClusterCreatorFactory.create()
    return cluster_creator.create_clusters(filename, 8)


def visualize():
    reduction_factory = ReductionMethodFactory()
    reduction_method = reduction_factory.create_method(cfg.REDUCTION_METHOD)
    all_clusters = create_clusters()
    logging.info(f'Clusters created')
    # plot_elbow_method(all_clusters)
    plot_clusters(all_clusters[0], reduction_method)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Clustering method: {cfg.CLUSTER_METHOD}')
    logging.info(f'Reduction method: {cfg.REDUCTION_METHOD}')
    visualize()
    logging.info("Done")
