import os
import subprocess
import shutil
import pandas as pd

from utils import get_root_path

ROOT_PATH = get_root_path()
DATA_PARENT_PATH = ROOT_PATH + '/data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104'
DATA_RAW_FILE_NAME = 'spikeprot0104.fasta'
SPLIT_FILES_DIR_NAME = 'split_data'
FILES_NAME_CORE = 'spikeprot_batch_data'

class DirHandler:
    split_files_dir_name = 'split_data'
    temporary_fasta_dir_name = 'temp_fasta'
    temporary_csv_dir_name = 'temp_csv'
    files_name_core = 'spikeprot_batch_data'

    @staticmethod
    def create_dirs():
        split_data_root_dir = DirHandler.__get_split_data_root_path()
        temporary_fasta_dir = DirHandler.get_temp_fasta_dir_path()
        temporary_csv_dir = DirHandler.get_temp_csv_dir_path()
        dir_paths = [split_data_root_dir, temporary_fasta_dir, temporary_csv_dir]
        for path in dir_paths:
            DirHandler.__create_dir(path)

    @staticmethod
    def __get_split_data_root_path():
        return f'{DATA_PARENT_PATH}/{DirHandler.split_files_dir_name}'

    @staticmethod
    def get_temp_fasta_dir_path():
        return f'{DirHandler.__get_split_data_root_path()}/temp_fasta'

    @staticmethod
    def get_temp_csv_dir_path():
        return f'{DirHandler.__get_split_data_root_path()}/temp_csv'

    @staticmethod
    def __create_dir(path):
        try:
            os.mkdir(path)
        except FileExistsError:
            print(f'{path} already exists')

    @staticmethod
    def delete_temp_fasta():
        shutil.rmtree(DirHandler.get_temp_fasta_dir_path(), ignore_errors=True)

    @staticmethod
    def delete_temp_csv():
        shutil.rmtree(DirHandler.get_temp_csv_dir_path(), ignore_errors=True)


class BatchSplitter:
    lines_num_each_file = 100
    max_num_of_files = 3

    @staticmethod
    def split_to_equal_files():
        max_num_iters = BatchSplitter.__get_iterations_num()
        start_line_idx = 1
        end_line_idx = BatchSplitter.lines_num_each_file
        filepath_with_name_core = DirHandler.get_temp_fasta_dir_path() + '/' + FILES_NAME_CORE
        raw_data_filepath = DATA_PARENT_PATH + '/' + DATA_RAW_FILE_NAME

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
        raw_data_filepath = DATA_PARENT_PATH + '/' + DATA_RAW_FILE_NAME
        num_lines_command = f'wc -l {raw_data_filepath}'
        output = subprocess.check_output(num_lines_command, shell=True, encoding='UTF-8')
        output = output.split()
        return output[0]


class CsvTransformer:
    @staticmethod
    def transform_files():
        files_names = CsvTransformer.__get_files_names()
        fasta_files_dir = DirHandler.get_temp_fasta_dir_path()
        csv_files_dir = DirHandler.get_temp_csv_dir_path()
        for name in files_names:
            read_file = pd.read_fwf(f'{fasta_files_dir}/{name}.fasta', header=None)
            description_df = CsvTransformer.__create_description_df(csv_file_df=read_file)
            sequence_df = CsvTransformer.__create_sequence_df(csv_file_df=read_file)
            result = pd.concat([description_df, sequence_df], axis=1, join='inner')
            result.to_csv(f'{csv_files_dir}/{name}.csv', index=False)

    @staticmethod
    def __get_files_names():
        split_files_dir_filepath = DirHandler.get_temp_fasta_dir_path()
        file_names = os.listdir(split_files_dir_filepath)
        return list(map(lambda x: x.rstrip('.fasta'), file_names))

    @staticmethod
    def __create_description_df(csv_file_df):
        description_df = csv_file_df.iloc[::2, :]
        description_df.columns = ['description']
        description_df.reset_index(drop=True, inplace=True)
        return description_df

    @staticmethod
    def __create_sequence_df(csv_file_df):
        sequence_df = csv_file_df.iloc[1::2, :]
        sequence_df.columns = ['sequence']
        sequence_df.reset_index(drop=True, inplace=True)
        return sequence_df


class BatchCleaner:
    @staticmethod
    def clean():
        pass


if __name__ == '__main__':
    DirHandler.create_dirs()
    BatchSplitter.split_to_equal_files()
    CsvTransformer.transform_files()
    # DirHandler.delete_temp_fasta()
