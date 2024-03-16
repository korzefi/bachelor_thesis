# author: Filip Korzeniewski


import os
import logging
from datetime import datetime


def get_root_path():
    root_path = os.path.dirname(os.path.abspath(__file__))
    splitted = root_path.split('/')[:-1]
    return '/'.join(splitted)


def create_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        logging.warning(f'{path} already exists')


def get_formatted_datetime():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d-%m-%Y_%H-%M")
    return formatted_datetime


def setup_logger(process_id=None, date=True, time=True):
    datefmt = "%Y-%m-%d" if date else ""
    timefmt = "%H:%M:%S" if time else ""
    datetimefmt = ""
    if date and time:
        datetimefmt = f"{datefmt} {timefmt}"
    else:
        datetimefmt = f"{datefmt}{timefmt}"

    if process_id is not None:
        process_id = f"_{process_id + 1}"
    else:
        process_id = ""

    logging.basicConfig(level=logging.INFO,
                        format=f"%(levelname)s{process_id} %(asctime)s: %(message)s",
                        datefmt=datetimefmt)


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