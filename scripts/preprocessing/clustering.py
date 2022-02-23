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
        for file in files:
            logging.info('transforming sequences for file: ')
            filepath = f'{cfg.DATA_PERIODS_PATH}/{file}'
            file_triplets = TripletMaker.createTriplets(filepath)
            print(VectorTransformer.transform(file_triplets, prot_vec))

    @staticmethod
    def __get_files_names():
        return os.listdir(cfg.DATA_PERIODS_PATH)


class TripletMaker:
    @staticmethod
    def createTriplets(filepath):
        df = pd.read_csv(filepath)
        seq_df = df['sequence']
        rows_num = seq_df.shape[0]
        logging.info(f'Transforming {rows_num} sequences:')
        return TripletMaker.__transform_seqs_into_triplets_vec(seq_df)

    @staticmethod
    def __transform_seqs_into_triplets_vec(seq_df) -> [[[]]]:
        """returns list of all sequences in form of triplets
            file - list of sequences
            sequence - list of triplets
            triplet - list of 3 amino acids"""
        result = []
        for seq in seq_df:
            seq_len = len(seq)
            triplets_num = seq_len - 2
            seq_triplets = [seq[i:i + 3] for i in range(triplets_num)]
            result.append(seq_triplets)
        return result


class VectorTransformer:
    @staticmethod
    def transform(file_triplets: [[[]]], prot_vec: pd.DataFrame):
        for sequence in file_triplets:
            for triplet in sequence:
                result = prot_vec.loc[prot_vec['words'] == triplet]
                print(result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ProtVecTransformer.transform_vector()
