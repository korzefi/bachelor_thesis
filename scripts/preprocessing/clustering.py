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
    def __create_df_template(df: pd.DataFrame):
        return df.drop(df.index, inplace=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ProtVecTransformer.transform_vector()
