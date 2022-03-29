import logging

from scripts.clustering.ClusterLinker import get_epitopes_list


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    get_epitopes_list()
