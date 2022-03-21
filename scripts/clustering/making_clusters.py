import logging
from config import MakingClusters as cfg
from ClusterCreator import ClusterCreator, ClusterMethodFactory


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    clusters = create_clusters()
    clusters = clusters[0]
    print(clusters['data'], clusters['labels'], clusters['centroids'])
    logging.info("Done")
