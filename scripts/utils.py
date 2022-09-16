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


def get_time_string(time):
    """
    Creates a string representation of minutes and seconds from the given time.
    """
    mins = time // 60
    secs = time % 60
    time_string = ''

    if mins < 10:
        time_string += '  '
    elif mins < 100:
        time_string += ' '

    time_string += '%dm ' % mins

    if secs < 10:
        time_string += ' '

    time_string += '%ds' % secs

    return time_string