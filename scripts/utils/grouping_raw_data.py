import os
import subprocess
import pandas as pd

from utils import get_root_path

ROOT_PATH = get_root_path()
DATA_PARENT_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104/'
DATA_RAW_FILE_NAME = 'spikeprot0104.fasta'
SPLIT_FILES_DIR_NAME = 'split_data'
FILES_NAME_CORE = 'spikeprot_batch_data'


class BatchSplitter:
    lines_num_each_file = 10000
    max_num_of_files = 3

    @staticmethod
    def split_to_equal_files():
        try:
            BatchSplitter.__create_dir_for_files()
            BatchSplitter.__split_to_equal_files()
        except FileExistsError:
            print("text files are already splitted")

    @staticmethod
    def __create_dir_for_files():
        os.mkdir(DATA_PARENT_PATH + SPLIT_FILES_DIR_NAME)


    @staticmethod
    def __split_to_equal_files():
        max_num_iters = BatchSplitter.__get_iterations_num()
        start_line_idx = 1
        end_line_idx = BatchSplitter.lines_num_each_file
        filepath_with_name_core = DATA_PARENT_PATH + SPLIT_FILES_DIR_NAME + '/' + FILES_NAME_CORE
        raw_data_filepath = DATA_PARENT_PATH + DATA_RAW_FILE_NAME

        for i in range(max_num_iters):
            copy_command = f"cat {raw_data_filepath} | sed -n '{start_line_idx},{end_line_idx}p' >> {filepath_with_name_core}-{start_line_idx}-{end_line_idx}.fasta"
            os.system(copy_command)
            start_line_idx += BatchSplitter.lines_num_each_file
            end_line_idx += BatchSplitter.lines_num_each_file

    @staticmethod
    def __get_iterations_num():
        lines_num = int(BatchSplitter.__get_num_of_lines())
        max_num_of_batches = lines_num // BatchSplitter.lines_num_each_file
        return min(BatchSplitter.max_num_of_files, max_num_of_batches)

    @staticmethod
    def __get_num_of_lines():
        raw_data_filepath = DATA_PARENT_PATH + DATA_RAW_FILE_NAME
        num_lines_command = f'wc -l {raw_data_filepath}'
        output = subprocess.check_output(num_lines_command, shell=True, encoding='UTF-8')
        output = output.split()
        return output[0]


class CsvTransformer:
    @staticmethod
    def transform_files():
        files_names = CsvTransformer.__get_files_names()
        split_files_dir_filepath = DATA_PARENT_PATH + SPLIT_FILES_DIR_NAME + '/'
        for name in files_names:
            read_file = pd.read_fwf(split_files_dir_filepath + name + '.txt')
            read_file.columns = ['column name']
            description_df = read_file.iloc[::2, :]
            sequence_df = read_file.iloc[1::2, :]
            frames = [description_df, sequence_df]
            result = pd.concat(frames)
            result.to_csv(split_files_dir_filepath + name + '.csv', index=False)
            # read_file.to_csv(split_files_dir_filepath + name + '.csv', index=False)

    @staticmethod
    def __get_files_names():
        split_files_dir_filepath = DATA_PARENT_PATH + SPLIT_FILES_DIR_NAME + '/'
        file_names = os.listdir(split_files_dir_filepath)
        return list(map(lambda x: x.rstrip('.txt'), file_names))

    @staticmethod
    def __transform_single_file():
        pass


if __name__ == '__main__':
    BatchSplitter.split_to_equal_files()
    # CsvTransformer.transform_files()
