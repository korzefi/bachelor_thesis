import os
import logging
import multiprocessing

import pandas as pd
from natsort import natsorted

import scripts.utils as utils
from scripts.preprocessing.config import Clustering, ClusterToProceed
from scripts.preprocessing import ClusterLinker, ClusterCentroidsDataCreator, EpitopeDataCreator

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


def get_filepath_cluster_dict_for_centroids() -> {}:
    file_cluster_dict = ClusterToProceed.FILES_CLUSTERS_NUM
    return {f'{Clustering.VECTOR_TEMP_DIR_PATH}/{file}': cluster for file, cluster in file_cluster_dict}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # filepath_clusters_data = get_filepath_cluster_dict_for_centroids()
    # ClusterCentroidsDataCreator.create_centroids_data(filepath_cluster_dict=filepath_clusters_data, modify_file=True)
    # ClusterLinker.link()
    EpitopeDataCreator.create_final_data()

