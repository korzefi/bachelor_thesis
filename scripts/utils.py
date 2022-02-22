import os


def get_root_path():
    root_path = os.path.dirname(os.path.abspath(__file__))
    splitted = root_path.split('/')[:-1]
    return '/'.join(splitted)
