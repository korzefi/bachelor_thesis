import scripts.utils as utils

from scripts.preprocessing.grouping_raw_data import prepare_files
from scripts.preprocessing.clustering import create_cluster_data
from scripts.training.training import train, test

if __name__ == '__main__':
    utils.setup_logger()
    # prepare_files()
    # create_cluster_data()
    # train()
    # test()



