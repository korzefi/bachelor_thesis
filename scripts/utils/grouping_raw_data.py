import os
import subprocess

DATA_PARENT_PATH = '../../data/covid-spike-GISAID/spikeprot0104.tar/spikeprot0104/'
DATA_RAW_FILE_NAME = 'spikeprot0104.fasta'
DATA_RAW_FILEPATH = DATA_PARENT_PATH + DATA_RAW_FILE_NAME

class BatchSplitter:
    splitted_data_dir_name = 'splitted_data'
    lines_num_each_file = 10000
    max_num_of_files = 3
    files_name_core = DATA_PARENT_PATH + splitted_data_dir_name + '/spikeprot_batch_data'

    @staticmethod
    def split_to_equal_files():
        BatchSplitter.__create_dir_for_files()

        max_num_iters = BatchSplitter.__get_iterations_num()
        start_line_idx = 1
        end_line_idx = BatchSplitter.lines_num_each_file

        for i in range(max_num_iters):
            copy_command = f"cat {DATA_RAW_FILEPATH} | sed -n '{start_line_idx},{end_line_idx}p' >> {BatchSplitter.files_name_core}-{start_line_idx}-{end_line_idx}.txt"
            os.system(copy_command)
            start_line_idx += BatchSplitter.lines_num_each_file
            end_line_idx += BatchSplitter.lines_num_each_file

    @staticmethod
    def __create_dir_for_files():
        try:
            os.mkdir(DATA_PARENT_PATH + BatchSplitter.splitted_data_dir_name)
        except FileExistsError:
            pass

    @staticmethod
    def __get_iterations_num():
        lines_num = int(BatchSplitter.__get_num_of_lines())
        max_num_of_batches = lines_num // BatchSplitter.lines_num_each_file
        return min(BatchSplitter.max_num_of_files, max_num_of_batches)

    @staticmethod
    def __get_num_of_lines():
        num_lines_command = f'wc -l {DATA_RAW_FILEPATH}'
        output = subprocess.check_output(num_lines_command, shell=True, encoding='UTF-8')
        output = output.split()
        return output[0]

    @staticmethod


if __name__ == '__main__':
    BatchSplitter.split_to_equal_files()
    print('DODAC GITA !!!!!!!!')