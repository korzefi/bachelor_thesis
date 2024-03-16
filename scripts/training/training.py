# author: Filip Korzeniewski

import scripts.utils as utils

from scripts.training.config import LoadingDatasetsConfig
from scripts.utils import get_root_path
from scripts.training.loading_datasets import load_dataset

from scripts.training import net_utils
from scripts.training import models

import sys
import logging
import torch
import numpy as np

root_path = get_root_path()
sys.path.append(root_path)


def train():
    torch.manual_seed(1)
    np.random.seed(1)

    logging.info('Loading datasets')
    train_trigram_vecs, train_labels = load_dataset(LoadingDatasetsConfig.TRAIN_DATASET_PATH)
    logging.info('Train dataset loaded')
    valid_trigram_vecs, valid_labels = load_dataset(LoadingDatasetsConfig.VALID_DATASET_PATH)
    logging.info('Test dataset loaded')

    # logistic regression - optional
    net_utils.logistic_regression(train_trigram_vecs, train_labels, valid_trigram_vecs, valid_labels)

    # X_train = torch.tensor(train_trigram_vecs, dtype=torch.float32)
    # Y_train = torch.tensor(train_labels, dtype=torch.int64)
    # X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
    # Y_test = torch.tensor(test_labels, dtype=torch.int64)
    #
    # # give weights for imbalanced dataset
    # _, counts = np.unique(Y_train, return_counts=True)
    # train_counts = max(counts)
    # train_imbalance = max(counts) / Y_train.shape[0]
    # _, counts = np.unique(Y_test, return_counts=True)
    # test_counts = max(counts)
    # test_imbalance = max(counts) / Y_test.shape[0]
    #
    # logging.info('Class imbalances:')
    # logging.info(' Training %.3f' % train_imbalance)
    # logging.info(' Testing  %.3f' % test_imbalance)
    #
    # input_dim = X_train.shape[2]
    # seq_length = X_train.shape[0]
    # # output dim is 2 because of 2 classes: 0 - dim[0] - non-mutated, dim[1] - mutated
    # output_dim = 2
    #
    # logging.info('Creating RNN model')
    # net = models.RnnModel(seq_length, input_dim, output_dim)
    #
    # logging.info('Training model')
    # net_utils.train_rnn(model=net, verify=False,
    #                     X=X_train, Y=Y_train,
    #                     X_test=X_test, Y_test=Y_test,
    #                     show_attention=False)
    #
    #
    # # logging.info('Creating classic attention model')
    # # net = models.AttnRnnModel(seq_length, input_dim, output_dim)
    # #
    # #
    # # net_utils.train_rnn(model=net, verify=False,
    # #                     X=X_train, Y=Y_train,
    # #                     X_test=X_test, Y_test=Y_test,
    # #                     show_attention=True)
    #
    #
    # # logging.info('Creating dual-attention model')
    # # net = models.DualAttnRnnModel(seq_length, input_dim, output_dim)
    # #
    # # logging.info('Training model')
    # # net_utils.train_rnn(model=net, verify=False,
    # #                     X=X_train, Y=Y_train,
    # #                     X_test=X_test, Y_test=Y_test,
    # #                     show_attention=True)


if __name__ == '__main__':
    utils.setup_logger()
    logging.basicConfig(level=logging.INFO)

    logging.info("Experimental results with attention models")
    train()
