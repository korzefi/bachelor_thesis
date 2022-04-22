import logging
import pandas as pd
import numpy as np

from natsort import natsort_keygen

from scripts.preprocessing.config import Clustering


class ClusterLinker:
    def link(self):
        logging.info('Linking clusters')
        filepath = Clustering.CLUSTERS_CENTROIDS_DATA_PATH
        df = pd.read_csv(filepath)
        sorted_df = self.__sort_centroids(df)
        next_clusters_links = self.__get_links(sorted_df)
        sorted_df.drop(['next_cluster'], axis=1, inplace=True, errors='ignore')
        sorted_df.insert(loc=2, column='next_cluster', value=next_clusters_links)
        sorted_df.to_csv(filepath, index=False)
        logging.info('clusters linked')

    def __sort_centroids(self, df):
        df.sort_values(by=['period', 'cluster'], key=natsort_keygen(), inplace=True)
        return df

    def __get_links(self, df):
        periods = df['period'].unique()
        cols_100dim = ['d' + str(i) for i in range(1, 101)]
        # dicts {row_index: linked_cluster_row_idx}
        forward_links = {}
        backward_links = {}
        if len(periods) < 2:
            ValueError('Not enough periods to link clusters')
        for i in range(len(periods) - 1):
            current_period = periods[i]
            next_period = periods[i + 1]
            forward_links.update(
                self.__link_clusters_idx_forward(df, current_period, next_period, cols_100dim))
            backward_links.update(
                self.__link_clusters_idx_backward(df, current_period, next_period, cols_100dim))
        combined_idx_links = self.__combine_idx_links(forward_links, backward_links)
        clusters_vals = self.__get_clusters_values(df['cluster'], combined_idx_links)
        return clusters_vals

    def __link_clusters_idx_forward(self, df, current_per, next_per, cols):
        links = {}
        for index1, per1 in df[df['period'] == current_per].iterrows():
            dist_min = {}
            for index2, per2 in df[df['period'] == next_per].iterrows():
                dist = self.__get_euclidean_dist(per1, per2, cols)
                dist_min[index2] = dist
            min_dist_idx = min(dist_min, key=dist_min.get)
            links[index1] = min_dist_idx
        return links

    def __link_clusters_idx_backward(self, df, current_per, next_per, cols):
        links = {}
        for index2, per2 in df[df['period'] == next_per].iterrows():
            dist_min = {}
            for index1, per1 in df[df['period'] == current_per].iterrows():
                dist = self.__get_euclidean_dist(per1, per2, cols)
                dist_min[index1] = dist
            min_dist_idx = min(dist_min, key=dist_min.get)
            links[index2] = min_dist_idx
        return links

    def __get_euclidean_dist(self, per1, per2, cols):
        per1 = np.array(per1[cols])
        per2 = np.array(per2[cols])
        dist = np.linalg.norm(per1 - per2)
        return dist

    def __combine_idx_links(self, forward_links, backward_links):
        links = {}
        self.__transform_idx_links(forward_links, backward_links)
        links.update(forward_links)
        for k, v in backward_links.items():
            if k in links:
                links[k].extend(backward_links[k])
            else:
                links[k] = backward_links[k]
        links = {k: list(set(v)) for k, v in links.items()}
        [v.sort() for v in links.values()]
        return links

    def __transform_idx_links(self, forward_links, backward_links):
        self.__transform_forward_links_to_lists(forward_links)
        self.__reverse_backward_links(backward_links)

    def __reverse_backward_links(self, backward_links):
        temp = {}
        for key, value in backward_links.items():
            if value in temp:
                temp[value].append(key)
            else:
                temp[value] = [key]
        backward_links.clear()
        backward_links.update(temp)

    def __transform_forward_links_to_lists(self, forward_links):
        temp = {k: [v] for k, v in forward_links.items()}
        forward_links.clear()
        forward_links.update(temp)

    def __get_clusters_values(self, clusters_data, combined_idx_links):
        clusters = self.__transform_to_clusters(clusters_data, combined_idx_links)
        cluster_numeric_vals = self.__fill_missing_clusters(clusters, len(clusters_data.index))
        cluster_vals = self.__clusters_links_to_str_format(cluster_numeric_vals)
        return cluster_vals

    def __transform_to_clusters(self, clusters_data, combined_idx_links):
        clusters = {}
        for idx, idx_links in combined_idx_links.items():
            clusters[idx] = [clusters_data.iloc[idx_link] for idx_link in idx_links]
        return clusters

    def __fill_missing_clusters(self, clusters, idx_num):
        for i in range(idx_num):
            if i not in clusters:
                clusters[i] = ''
        return list(clusters.values())

    def __clusters_links_to_str_format(self, cluster_numeric_vals):
        # format is 'cluster1-cluster2...-clustern' ex. '0-2-3' or '' for empty
        # clusters_data = [str(cluster) for clusters in cluster_numeric_vals for cluster in clusters]
        for i in range(len(cluster_numeric_vals)):
            cluster_numeric_vals[i] = list(map(lambda x: str(x), cluster_numeric_vals[i]))
        return list(map(lambda clusters: '-'.join(clusters), cluster_numeric_vals))


def link():
    linker = ClusterLinker()
    linker.link()
