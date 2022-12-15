import logging

import pandas as pd
from sklearn.cluster import KMeans

from scripts.clustering.config import MakingClusters as cfg
from scripts.clustering.config import KMeansConfig


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
        all_jobs_activated = -1
        clf = KMeans(n_clusters=n_clusters,
                     n_init=KMeansConfig.N_ITERATIONS_INIT,
                     max_iter=KMeansConfig.MAX_ITER)
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

    def create_clusters(self, filepath, use_range=True, n_clusters=None):
        filename = self.__get_filename(filepath)
        df = pd.read_csv(filepath)
        logging.info(f'Clustering sequences, file: {filename}')
        if (n_clusters is None) or (use_range is True):
            return self.__create_plenty_clusters_versions(df)
        else:
            return self.__create_specific_clusters_variant(df, n_clusters)

    def __create_plenty_clusters_versions(self, df) -> [{}]:
        for n_clusters in range(cfg.INIT_N_CLUSTERS, cfg.END_N_CLUSTERS + 1):
            logging.info(f'Clustering sequences, clusters num: {n_clusters}')
            cluster = self.cluster_method.compute(df.to_numpy(), n_clusters)
            self.__all_clusters.append(cluster)
        return self.__all_clusters

    def __create_specific_clusters_variant(self, df, n_clusters: int) -> {}:
        logging.info(f'Clustering sequences, clusters num: {n_clusters}')
        cluster = self.cluster_method.compute(df.to_numpy(), n_clusters)
        return cluster

    def __get_filename(self, filepath):
        return filepath.split('/')[-1]
