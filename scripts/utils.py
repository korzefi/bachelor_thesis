import os
import logging


def get_root_path():
    root_path = os.path.dirname(os.path.abspath(__file__))
    splitted = root_path.split('/')[:-1]
    return '/'.join(splitted)


def create_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        logging.warning(f'{path} already exists')
