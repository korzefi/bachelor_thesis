import pandas as pd
import os

from scripts.preprocessing.config import Clustering
from scripts.clustering.making_clusters import ClusterCreatorFactory


class ClusterDataCreator:
    def create_centroids_data(self, filepath, n_clusters, modify_file=False):
        cluster_creator = ClusterCreatorFactory.create()
        clusters = cluster_creator.create_clusters(filepath=filepath, use_range=False, n_clusters=n_clusters)
        filename = filepath.split('/')[-1]
        if modify_file:
            self.__add_cluster_column(filename, clusters['labels'])
        self.__add_centroids_data(filename, clusters['centroids'])

    def __add_cluster_column(self, filename, labels):
        filepath = f'{Clustering.DATA_PERIODS_UNIQUE_PATH}/{filename}'
        df = pd.read_csv(filepath)
        df.drop(['cluster'], axis=1, inplace=True, errors='ignore')
        df.insert(loc=0, column='cluster', value=labels)
        df.to_csv(filepath, index=False)

    def __add_centroids_data(self, filename, centroids):
        self.__create_centroid_file_if_not_exist()
        period = filename[:-4]
        filepath = Clustering.CLUSTERS_CENTROIDS_DATA_PATH
        columns = self.__create_columns_headers()
        df = pd.read_csv(filepath)
        for cluster_num, centroid in enumerate(centroids):
            data = [period, cluster_num]
            data += centroid.tolist()
            row = pd.DataFrame([data], columns=columns)
            df = pd.concat([df, row], ignore_index=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(filepath, index=False)

    def __create_centroid_file_if_not_exist(self):
        filepath = Clustering.CLUSTERS_CENTROIDS_DATA_PATH
        if os.path.exists(filepath):
            return
        columns = ClusterDataCreator.__create_columns_headers(self)
        df = pd.DataFrame(columns=columns)
        df.to_csv(filepath, index=False, header=True)

    def __create_columns_headers(self):
        columns = ['period', 'cluster']
        cols_100dim = ['d' + str(i) for i in range(1, 101)]
        columns += cols_100dim
        return columns


def create_centroids_data(filepath_cluster_dict: {}, modify_file=False):
    creator = ClusterDataCreator()
    for file, cluster in filepath_cluster_dict.items():
        creator.create_centroids_data(file, creator, modify_file)
