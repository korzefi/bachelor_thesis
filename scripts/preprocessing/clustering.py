import os
import logging
import multiprocessing
from itertools import product

import pandas as pd
from natsort import natsorted
from natsort import natsort_keygen
import numpy as np

import scripts.utils as utils
from scripts.preprocessing.config import Clustering
from scripts.clustering.making_clusters import ClusterCreatorFactory

LOGGING_PROCESSES_ENABLED = True


class ProtVecTransformer:
    @staticmethod
    def transform_vector(file_num):
        if LOGGING_PROCESSES_ENABLED:
            logging.basicConfig(level=logging.INFO)
        logging.info('Transforming sequences')
        files = ProtVecTransformer.__get_files_names()
        files = natsorted(files)
        prot_vec = pd.read_csv(Clustering.PROT_VEC_PATH)
        utils.create_dir(Clustering.VECTOR_TEMP_DIR_PATH)
        files = files[file_num:file_num + 1]
        for file in files:
            logging.info(f'transforming sequences for file: {file}')
            filepath = f'{Clustering.DATA_PERIODS_UNIQUE_PATH}/{file}'
            file_triplets = TripletMaker.createTriplets(filepath)
            file_vec = VectorTransformer.transform(file_triplets, prot_vec)
            file_vec.to_csv(f'{Clustering.VECTOR_TEMP_DIR_PATH}/{file}', index=False)
        logging.info('Done')

    @staticmethod
    def __get_files_names():
        return os.listdir(Clustering.DATA_PERIODS_UNIQUE_PATH)


class TripletMaker:
    @staticmethod
    def createTriplets(filepath) -> [[]]:
        df = pd.read_csv(filepath)
        seq_df = df['sequence']
        rows_num = seq_df.shape[0]
        logging.info(f'Transforming {rows_num} sequences:')
        return TripletMaker.__transform_seqs_into_triplets_vec(seq_df)

    @staticmethod
    def __transform_seqs_into_triplets_vec(seq_df) -> [[]]:
        """returns list of all sequences in form of triplets
            file - list of sequences
            sequence - list of triplets
            triplet - string of 3 amino acids"""
        result = []
        for seq in seq_df:
            seq_len = len(seq)
            triplets_num = seq_len - 2
            seq_triplets = [seq[i:i + 3] for i in range(triplets_num)]
            result.append(seq_triplets)
        return result


class VectorTransformer:
    VEC_DECIMAL_PLACES = 8

    @staticmethod
    def transform(file_triplets: [[]], prot_vec: pd.DataFrame) -> pd.DataFrame:
        """takes list of sequences of triplets
            returns list of sequences, where each sequence is a vector of 100-dim"""
        result = VectorTransformer.__create_df_template(prot_vec)
        result.drop('words', inplace=True, axis=1)
        sequence_counter = 0
        for sequence in file_triplets:
            sequence_counter += 1
            logging.info(f'sequence {sequence_counter} of {len(file_triplets)} is being transformed...')
            seq_result = VectorTransformer.__create_df_template(prot_vec)
            for triplet in sequence:
                found = prot_vec.loc[prot_vec['words'] == triplet]
                seq_result = pd.concat([seq_result, found])
            seq_vector = VectorTransformer.__add_embedded_vectors(seq_result)
            result = pd.concat([result, seq_vector], ignore_index=True)
        VectorTransformer.__round_vector_df(result, decimals=VectorTransformer.VEC_DECIMAL_PLACES)
        return result

    @staticmethod
    def __add_embedded_vectors(seq_vectors: pd.DataFrame) -> pd.DataFrame:
        seq_vectors.drop('words', inplace=True, axis=1)
        seq_vectors.reset_index(drop=True, inplace=True)
        result = pd.DataFrame()
        for column in seq_vectors:
            result.loc[1, column] = seq_vectors[column].sum()
        return result

    @staticmethod
    def __round_vector_df(df, decimals):
        for column in df:
            df[column] = df[column].round(decimals=decimals)

    @staticmethod
    def __create_df_template(df: pd.DataFrame):
        return df.drop(df.index, inplace=False)


def transform_vectors_multiprocess(first_file_num, last_file_num):
    processes = []
    for i in range(first_file_num, last_file_num + 1):
        p = multiprocessing.Process(target=ProtVecTransformer.transform_vector, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


class ClusterDataCreator:
    @staticmethod
    def create_centroids_data(filepath, n_clusters, modify_file=False):
        cluster_creator = ClusterCreatorFactory.create()
        clusters = cluster_creator.create_clusters(filepath=filepath, use_range=False, n_clusters=n_clusters)
        filename = filepath.split('/')[-1]
        if modify_file:
            ClusterDataCreator.__add_cluster_column(filename, clusters['labels'])
        ClusterDataCreator.__add_centroids_data(filename, clusters['centroids'])

    @staticmethod
    def __add_cluster_column(filename, labels):
        filepath = f'{Clustering.DATA_PERIODS_UNIQUE_PATH}/{filename}'
        df = pd.read_csv(filepath)
        df.drop(['cluster'], axis=1, inplace=True, errors='ignore')
        df.insert(loc=0, column='cluster', value=labels)
        df.to_csv(filepath, index=False)

    @staticmethod
    def __add_centroids_data(filename, centroids):
        ClusterDataCreator.__create_centroid_file_if_not_exist()
        period = filename[:-4]
        filepath = Clustering.CLUSTERS_CENTROIDS_DATA_PATH
        columns = ClusterDataCreator.__create_columns_headers()
        df = pd.read_csv(filepath)
        for cluster_num, centroid in enumerate(centroids):
            data = [period, cluster_num]
            data += centroid.tolist()
            row = pd.DataFrame([data], columns=columns)
            df = pd.concat([df, row], ignore_index=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(filepath, index=False)

    @staticmethod
    def __create_centroid_file_if_not_exist():
        filepath = Clustering.CLUSTERS_CENTROIDS_DATA_PATH
        if os.path.exists(filepath):
            return
        columns = ClusterDataCreator.__create_columns_headers()
        df = pd.DataFrame(columns=columns)
        df.to_csv(filepath, index=False, header=True)

    @staticmethod
    def __create_columns_headers():
        columns = ['period', 'cluster']
        cols_100dim = ['d' + str(i) for i in range(1, 101)]
        columns += cols_100dim
        return columns


class ClusterLinker:
    @staticmethod
    def link():
        logging.info('Linking clusters')
        filepath = Clustering.CLUSTERS_CENTROIDS_DATA_PATH
        df = pd.read_csv(filepath)
        sorted_df = ClusterLinker.__sort_centroids(df)
        next_clusters_links = ClusterLinker.__get_links(df)
        df.drop(['next_cluster'], axis=1, inplace=True, errors='ignore')
        df.insert(loc=2, column='next_cluster', value=next_clusters_links)
        df.to_csv(filepath, index=False)
        logging.info('clusters linked')

    @staticmethod
    def __sort_centroids(df):
        df.sort_values(by=['period', 'cluster'], key=natsort_keygen(), inplace=True)
        return df

    @staticmethod
    def __get_links(df):
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
                ClusterLinker.__link_clusters_idx_forward(df, current_period, next_period, cols_100dim))
            backward_links.update(
                ClusterLinker.__link_clusters_idx_backward(df, current_period, next_period, cols_100dim))
        combined_idx_links = ClusterLinker.__combine_idx_links(forward_links, backward_links)
        clusters_vals = ClusterLinker.__get_clusters_values(df['cluster'], combined_idx_links)
        return clusters_vals

    @staticmethod
    def __link_clusters_idx_forward(df, current_per, next_per, cols):
        links = {}
        for index1, per1 in df[df['period'] == current_per].iterrows():
            dist_min = {}
            for index2, per2 in df[df['period'] == next_per].iterrows():
                dist = ClusterLinker.__get_euclidean_dist(per1, per2, cols)
                dist_min[index2] = dist
            min_dist_idx = min(dist_min, key=dist_min.get)
            links[index1] = min_dist_idx
        return links

    @staticmethod
    def __link_clusters_idx_backward(df, current_per, next_per, cols):
        links = {}
        for index2, per2 in df[df['period'] == next_per].iterrows():
            dist_min = {}
            for index1, per1 in df[df['period'] == current_per].iterrows():
                dist = ClusterLinker.__get_euclidean_dist(per1, per2, cols)
                dist_min[index1] = dist
            min_dist_idx = min(dist_min, key=dist_min.get)
            links[index2] = min_dist_idx
        return links

    @staticmethod
    def __get_euclidean_dist(per1, per2, cols):
        per1 = np.array(per1[cols])
        per2 = np.array(per2[cols])
        dist = np.linalg.norm(per1 - per2)
        return dist

    @staticmethod
    def __combine_idx_links(forward_links, backward_links):
        links = {}
        ClusterLinker.__transform_idx_links(forward_links, backward_links)
        links.update(forward_links)
        for k, v in backward_links.items():
            if k in links:
                links[k].extend(backward_links[k])
            else:
                links[k] = backward_links[k]
        links = {k: list(set(v)) for k, v in links.items()}
        [v.sort() for v in links.values()]
        return links

    @staticmethod
    def __transform_idx_links(forward_links, backward_links):
        ClusterLinker.__transform_forward_links_to_lists(forward_links)
        ClusterLinker.__reverse_backward_links(backward_links)

    @staticmethod
    def __reverse_backward_links(backward_links):
        temp = {}
        for key, value in backward_links.items():
            if value in temp:
                temp[value].append(key)
            else:
                temp[value] = [key]
        backward_links.clear()
        backward_links.update(temp)

    @staticmethod
    def __transform_forward_links_to_lists(forward_links):
        temp = {k: [v] for k, v in forward_links.items()}
        forward_links.clear()
        forward_links.update(temp)

    @staticmethod
    def __get_clusters_values(clusters_data, combined_idx_links):
        clusters = ClusterLinker.__transform_to_clusters(clusters_data, combined_idx_links)
        cluster_numeric_vals = ClusterLinker.__fill_missing_clusters(clusters, len(clusters_data.index))
        cluster_vals = ClusterLinker.__clusters_links_to_str_format(cluster_numeric_vals)
        return cluster_vals

    @staticmethod
    def __transform_to_clusters(clusters_data, combined_idx_links):
        clusters = {}
        for idx, idx_links in combined_idx_links.items():
            clusters[idx] = [clusters_data.iloc[idx_link] for idx_link in idx_links]
        return clusters

    @staticmethod
    def __fill_missing_clusters(clusters, idx_num):
        for i in range(idx_num):
            if i not in clusters:
                clusters[i] = ''
        return list(clusters.values())

    @staticmethod
    def __clusters_links_to_str_format(cluster_numeric_vals):
        # format is 'cluster1-cluster2...-clustern' ex. '0-2-3' or '' for empty
        # clusters_data = [str(cluster) for clusters in cluster_numeric_vals for cluster in clusters]
        for i in range (len(cluster_numeric_vals)):
            cluster_numeric_vals[i] = list(map(lambda x: str(x), cluster_numeric_vals[i]))
        return list(map(lambda clusters: '-'.join(clusters), cluster_numeric_vals))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # file = '2021-1.csv'
    # n_clusters = 3
    # filepath = f'{Clustering.VECTOR_TEMP_DIR_PATH}/{file}'
    # ClusterDataCreator.create_centroids_data(filepath, n_clusters, modify_file=True)
    ClusterLinker.link()
