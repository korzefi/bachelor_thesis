import os
from config import Clustering as cfg


def get_files_names():
    return os.listdir(cfg.DATA_PERIODS_PATH)


def create_clusters():
    file_names = get_files_names()
    for file in file_names:
        print(file)


if __name__ == '__main__':
    create_clusters()
