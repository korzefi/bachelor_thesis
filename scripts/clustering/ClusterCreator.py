import logging
import pandas as pd
from sklearn.cluster import KMeans

from config import MakingClusters as cfg
from config import KMeansConfig


class ClusterMethod:
    def compute(self, period_vec_data, n_clusters) -> {}:
        return self._execute(period_vec_data, n_clusters)

    def _execute(self, period_vec_data, dims) -> {}:
        return {}


class KMeansMethod(ClusterMethod):
    def __init__(self):
        super().__init__()

    def _execute(self, period_vec_data, n_clusters) -> {}:
        # by the supplementary it is changed by user in each year using Elbow method
        clf = KMeans(n_clusters=n_clusters, n_init=KMeansConfig.N_ITERATIONS_INIT)
        clf.fit(period_vec_data)
        labels = clf.labels_
        centroids = clf.cluster_centers_
        distortion = clf.inertia_

        return {'data': period_vec_data,
                'labels': labels,
                'centroids': centroids,
                'distortion': distortion,
                'n_clusters': n_clusters}


class ClusterMethodFactory:
    def __init__(self):
        self.methods = {'KMeans': KMeansMethod}

    def create_method(self, method_name):
        return self.methods[method_name]()


class ClusterCreator:
    def __init__(self, cluster_method: ClusterMethod):
        self.cluster_method = cluster_method
        self.__all_clusters = []

    def create_clusters(self, filename, n_clusters=None):
        filepath = self.__get_filepath(filename)
        df = pd.read_csv(filepath)
        logging.info(f'Clustering sequences, file: {filename}')
        if n_clusters is None:
            return self.__create_plenty_clusters_versions(df)
        else:
            return self.__create_specific_clusters_variant(df, n_clusters)

    def __create_plenty_clusters_versions(self, df):
        for n_clusters in range(cfg.INIT_N_CLUSTERS, cfg.END_N_CLUSTERS + 1):
            logging.info(f'Clustering sequences, clusters num: {n_clusters}')
            cluster = self.cluster_method.compute(df.to_numpy(), n_clusters)
            self.__all_clusters.append(cluster)
        return self.__all_clusters

    def __create_specific_clusters_variant(self, df, n_clusters: int):
        logging.info(f'Clustering sequences, clusters num: {n_clusters}')
        cluster = self.cluster_method.compute(df.to_numpy(), n_clusters)
        self.__all_clusters.append(cluster)
        return self.__all_clusters

    def __get_filepath(self, filename):
        return f'{cfg.EMBEDDED_DATA_PATH}/{filename}'
