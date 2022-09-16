from scripts.training.config import LoadingDatasetsConfig

import pandas as pd
import ast
import numpy as np


def read_trigram_vecs():
    prot_vec_path = LoadingDatasetsConfig.PROT_VEC_PATH

    # reads values separately
    # df = pd.read_csv(prot_vec_path, delimiter='\t')
    df = pd.read_csv(prot_vec_path)

    # gets all trigrams names
    trigrams = list(df['words'])

    # gets indexes of trigrams, so trigram_to_idx is {trigram_name: index}
    trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}

    # group of vecs from dataframe: takes all rows(:) and columns that are not words (all d columns)
    # transforms it to values; these are all those numbers
    trigram_vecs = df.loc[:, df.columns != 'words'].values

    # returns dict of all {trigram: index} and the values for d1-d100 columns
    return trigram_to_idx, trigram_vecs


def map_idxs_to_vecs(nested_idx_list, idx_to_vec):
    """
    Takes a nested list of indexes and maps them to their trigram vec (np array).
    """

    def mapping(idx):
        if isinstance(idx, int):
            return idx_to_vec[idx]
        elif isinstance(idx, list):
            return list(map(mapping, idx))
        else:
            raise TypeError('Expected nested list of ints, but encountered {} in recursion.'.format(type(idx)))

    return list(map(mapping, nested_idx_list))


def load_dataset(filepath):
    _, trigram_vecs_data = read_trigram_vecs()

    dataset_path = filepath
    df = pd.read_csv(dataset_path)

    # TODO: for sure .values here? - .values orders it in columns instead of rows (without .values)
    labels = df['y'].values

    # 3-dim array: [sample][period][trigram_as_string]
    trigram_idx_strings = df.loc[:, df.columns != 'y'].values

    # parse trigram_as_string to lists of 3 values - [sample][period][trigrams - 3 values]
    parsed_trigram_idxs = [list(map(lambda x: ast.literal_eval(x), example)) for example in trigram_idx_strings]

    # change 3 values indexes to 3 vectors of d1-d100 vecs
    # now there are 4 dims [sample][period][trigram_num (0, 1 or 2)][d1-d100 list of values]
    trigram_vecs = np.array(map_idxs_to_vecs(parsed_trigram_idxs, trigram_vecs_data))

    # Sum trigram vecs - [sample][period][summed trigrams by each d[i]]
    trigram_vecs = np.sum(trigram_vecs, axis=2)

    # move period dim as first so that [period][sample][d1-d100 dims]
    trigram_vecs = np.moveaxis(trigram_vecs, 1, 0)

    return trigram_vecs, labels


if __name__ == '__main__':
    trigram_vecs, labels = load_dataset(f'{LoadingDatasetsConfig.DATASETS_DIRPATH}/train-15-09-22.csv')
    pass
