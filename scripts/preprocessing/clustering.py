import os
import pandas as pd
import logging
from config import Clustering as cfg


class ProtVecTransformer:
    @staticmethod
    def transform_vector():
        logging.info('Transforming sequences')
        files = ProtVecTransformer.__get_files_names()
        files.sort()
        prot_vec = pd.read_csv(cfg.PROT_VEC_PATH)
        files_vectors = {}
        for file in files:
            logging.info(f'transforming sequences for file: {file}')
            filepath = f'{cfg.DATA_PERIODS_PATH}/{file}'
            file_triplets = TripletMaker.createTriplets(filepath)
            file_vec = VectorTransformer.transform(file_triplets, prot_vec)
            files_vectors[file] = file_vec

    @staticmethod
    def __get_files_names():
        return os.listdir(cfg.DATA_PERIODS_PATH)


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
    @staticmethod
    def transform(file_triplets: [[]], prot_vec: pd.DataFrame) -> [[]]:
        """takes list of sequences of triplets
            returns list of sequences, where each sequence is a vector of 100-dim"""
        result = []
        for sequence in file_triplets:
            seq_result = prot_vec.drop(prot_vec.index, inplace=False)
            for triplet in sequence:
                found = prot_vec.loc[prot_vec['words'] == triplet]
                seq_result = pd.concat([seq_result, found])
            seq_vector = VectorTransformer.__add_embedded_vectors(seq_result)
            result.append(seq_vector)
        return result

    @staticmethod
    def __add_embedded_vectors(vectors_list: [[]]):
        vectors_list.drop('words', inplace=True, axis=1)
        return [sum(x) for x in zip(vectors_list)]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ProtVecTransformer.transform_vector()
