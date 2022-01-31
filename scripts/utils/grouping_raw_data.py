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
DIVISION_TECHNIQUE = 'year'


class DirHandler:
    split_files_dir_name = 'split_data'
    temporary_fasta_dir_name = 'temp_fasta'
    temporary_csv_dir_name = 'temp_csv'
    period_root_dir_name = 'periods'
    files_name_core = 'spikeprot_batch_data'
    first_year_dir = '2019'
    last_year_dir = '2022'

    @staticmethod
    def create_dirs():
        split_data_root_dir = DirHandler.__get_split_data_root_path()
        temporary_fasta_dir = DirHandler.get_temp_fasta_dir_path()
        temporary_csv_dir = DirHandler.get_temp_csv_dir_path()
        periods_dirs = DirHandler.get_period_dirs()
        dir_paths = [split_data_root_dir, temporary_fasta_dir, temporary_csv_dir] + periods_dirs
        for path in dir_paths:
            DirHandler.__create_dir(path)

    @staticmethod
    def __get_split_data_root_path():
        return f'{DATA_PARENT_PATH}/{DirHandler.split_files_dir_name}'

    @staticmethod
    def get_temp_fasta_dir_path():
        return f'{DirHandler.__get_split_data_root_path()}/{DirHandler.temporary_fasta_dir_name}'

    @staticmethod
    def get_temp_csv_dir_path():
        return f'{DirHandler.__get_split_data_root_path()}/{DirHandler.temporary_csv_dir_name}'

    @staticmethod
    def __create_dir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            print(f'{path} already exists')

    @staticmethod
    def delete_temp_fasta():
        shutil.rmtree(DirHandler.get_temp_fasta_dir_path(), ignore_errors=True)

    @staticmethod
    def delete_temp_csv():
        shutil.rmtree(DirHandler.get_temp_csv_dir_path(), ignore_errors=True)

    @staticmethod
    def get_files_names(dir_path, extension):
        file_names = os.listdir(dir_path)
        return list(map(lambda x: x.rstrip('.' + extension), file_names))

    @staticmethod
    def get_csv_files_paths():
        csv_dir_path = DirHandler.get_temp_csv_dir_path()
        files_names = DirHandler.get_files_names(csv_dir_path, extension='.csv')
        return list(map(lambda x: csv_dir_path + '/' + x + '.csv', files_names))

    @staticmethod
    def get_period_dirs():
        root_dir = DirHandler.get_period_root_dir()
        periods = [f'{root_dir}/{str(year)}' for year in range(int(DirHandler.first_year_dir), int(DirHandler.last_year_dir)+1)]
        if DIVISION_TECHNIQUE == 'month':
            periods = DirHandler.__add_months(periods)
            return periods
        if DIVISION_TECHNIQUE == 'quarter':
            periods = DirHandler.__add_quarters(periods)
            return periods
        return periods

    @staticmethod
    def get_period_root_dir():
        return f'{DirHandler.__get_split_data_root_path()}/{DirHandler.period_root_dir_name}'

    @staticmethod
    def __add_months(period_years):
        months = range(1, 12+1)
        months = map(lambda x: str(x), months)
        return [f'{year}/{month}' for month in months for year in period_years]

    @staticmethod
    def __add_quarters(period_years):
        quarters = range(1, 4 + 1)
        quarters = map(lambda x: str(x), quarters)
        return [f'{year}/{quarter}' for quarter in quarters for year in period_years]



class BatchSplitter:
    lines_num_each_file = 100
    max_num_of_files = 3
    start_line_idx = 1

    @staticmethod
    def split_to_equal_files():
        max_num_iters = BatchSplitter.__get_iterations_num()
        end_line_idx = BatchSplitter.lines_num_each_file
        filepath_with_name_core = DirHandler.get_temp_fasta_dir_path() + '/' + FILES_NAME_CORE
        raw_data_filepath = DATA_PARENT_PATH + '/' + DATA_RAW_FILE_NAME

        for i in range(max_num_iters):
            copy_command = ['cat', raw_data_filepath, '|', 'sed', '-n',
                            f'{BatchSplitter.start_line_idx},{end_line_idx}p', '>>',
                            f"{filepath_with_name_core}-{BatchSplitter.start_line_idx}-{end_line_idx}.fasta"]
            os.system(' '.join(copy_command))
            BatchSplitter.start_line_idx += BatchSplitter.lines_num_each_file
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
        fasta_files_dir = DirHandler.get_temp_fasta_dir_path()
        files_names = DirHandler.get_files_names(dir_path=fasta_files_dir, extension='fasta')
        csv_files_dir = DirHandler.get_temp_csv_dir_path()
        for name in files_names:
            read_file = pd.read_fwf(f'{fasta_files_dir}/{name}.fasta', header=None)
            description_df = CsvTransformer.__create_description_df(csv_file_df=read_file)
            sequence_df = CsvTransformer.__create_sequence_df(csv_file_df=read_file)
            result = pd.concat([description_df, sequence_df], axis=1, join='inner')
            result.to_csv(f'{csv_files_dir}/{name}.csv', index=False)

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
    min_len = 1260

    @staticmethod
    def clean():
        files_paths = DirHandler.get_csv_files_paths()
        for file in files_paths:
            df = pd.read_csv(file)
            df = BatchCleaner.__remove_ambiguous(df)
            df = BatchCleaner.__remove_short(df)
            df = BatchCleaner.__filter_description(df)
            df = BatchCleaner.__remove_duplicates(df)
            df.to_csv(file, index=False)

    @staticmethod
    def __remove_ambiguous(df):
        ambiguous_aminos = ['B', 'J', 'Z', 'X', '-']
        df = df[~df['sequence'].str.contains('|'.join(ambiguous_aminos))]
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def __remove_short(df):
        end_sign = '*'
        df = df[df['sequence'].str.endswith(end_sign)]
        df = df[df['sequence'].str.len() > BatchCleaner.min_len]
        return df

    @staticmethod
    def __filter_description(df):
        splitted_desc = df['description'].str.split(pat='|', expand=True, n=3)
        splitted_desc.columns = ['gene', 'isolate_name', 'timestamp', 'rest']
        splitted_desc.drop(columns=['gene', 'rest'], inplace=True)
        splitted_desc = BatchCleaner.__adapt_day_format(splitted_desc)
        df.drop(columns='description', inplace=True)
        df = pd.concat([splitted_desc, df], axis=1, join='inner')
        return df

    @staticmethod
    def __adapt_day_format(df):
        days = pd.to_numeric(df['timestamp'].str[-2:]) + 1
        df['timestamp'] = df['timestamp'].str[:-2] + days.astype(str)
        return df

    @staticmethod
    def __remove_duplicates(df):
        df.drop_duplicates(subset=['isolate_name'], inplace=True)
        return df


class PeriodSorter:
    @staticmethod
    def divide():
        files_paths = DirHandler.get_csv_files_paths()
        for file in files_paths:
            df = pd.read_csv(file)
            PeriodSorter.__sort(df)
            df.to_csv(file, index=False)
            PeriodSorter.__divide(df)

    @staticmethod
    def __sort(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d', errors='coerce')
        df.sort_values(by='timestamp', inplace=True)
        df.dropna(subset='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def __divide(df):
        types = {'year': PeriodSorter.__divide_by_year,
                 'quarter': PeriodSorter.__divide_by_quarter,
                 'month': PeriodSorter.__divide_by_month}
        types[DIVISION_TECHNIQUE](df)

    @staticmethod
    def __divide_by_year(df):
        pass

    @staticmethod
    def __divide_by_quarter(df):
        pass

    @staticmethod
    def __divide_by_month(df):
        pass


if __name__ == '__main__':
    DirHandler.create_dirs()
    # BatchSplitter.split_to_equal_files()
    # CsvTransformer.transform_files()
    # DirHandler.delete_temp_fasta()
    # BatchCleaner.clean()
    # PeriodSorter.divide()
