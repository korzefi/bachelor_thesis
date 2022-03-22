import logging
import pandas as pd
from sklearn.cluster import KMeans

from config import MakingClusters as cfg
from config import KMeansConfig


class ClusterMethod:
    def __init__(self):
        self.clusters = []

    def compute(self, period_vec_data):
        self._execute(period_vec_data)

    def _execute(self, period_vec_data):
        pass


class KMeansMethod(ClusterMethod):
    def __init__(self):
        super().__init__()

    def _execute(self, period_vec_data):
        # by the supplementary it is changed by user in each year using Elbow method
        clf = KMeans(n_clusters=KMeansConfig.N_CLUSTERS)
        clf.fit(period_vec_data)
        labels = clf.labels_
        centroids = clf.cluster_centers_

        self.clusters.append({'data': period_vec_data, 'labels': labels, 'centroids': centroids})


class ClusterMethodFactory:
    def __init__(self):
        self.methods = {'KMeans': KMeansMethod}

    def create_method(self, method_name):
        return self.methods[method_name]()


class ClusterCreator:
    def __init__(self, cluster_method: ClusterMethod):
        self.cluster_method = cluster_method

    def create_clusters(self, filename):
        logging.info(f'Clustering sequences, file: {filename}')
        filepath = self.__get_filepath(filename)
        df = pd.read_csv(filepath)
        self.cluster_method.compute(df.to_numpy())
        return self.cluster_method.clusters

    def __get_filepath(self, filename):
        return f'{cfg.EMBEDDED_DATA_PATH}/{filename}'
