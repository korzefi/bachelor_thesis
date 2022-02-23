import os
import subprocess
import shutil
import pandas as pd
import logging

from config import GroupingRawData as cfg


class DirHandler:
    temporary_fasta_dir_name = 'temp_fasta'
    temporary_csv_dir_name = 'temp_csv'
    period_root_dir_name = 'periods'
    files_name_core = 'spikeprot_batch_data'

    @staticmethod
    def create_dirs():
        logging.info("Creating dirs")
        split_data_root_dir = DirHandler.__get_split_data_root_path()
        temporary_fasta_dir = DirHandler.get_temp_fasta_dir_path()
        temporary_csv_dir = DirHandler.get_temp_csv_dir_path()
        periods_dir = DirHandler.get_periods_dir()
        dir_paths = [split_data_root_dir, temporary_fasta_dir, temporary_csv_dir, periods_dir]
        for path in dir_paths:
            DirHandler.__create_dir(path)

    @staticmethod
    def __get_split_data_root_path():
        return f'{cfg.DATA_PARENT_PATH}/{cfg.SPLIT_FILES_DIR_NAME}'

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
            logging.warning(f'{path} already exists')

    @staticmethod
    def delete_temp_fasta():
        logging.info('Removing dir with generated .fasta files')
        shutil.rmtree(DirHandler.get_temp_fasta_dir_path(), ignore_errors=True)

    @staticmethod
    def delete_temp_csv():
        logging.info('Removing dir with generated .csv files')
        shutil.rmtree(DirHandler.get_temp_csv_dir_path(), ignore_errors=True)

    @staticmethod
    def get_files_names(dir_path, extension):
        file_names = os.listdir(dir_path)
        file_names.sort()
        return list(map(lambda x: x.rstrip('.' + extension), file_names))

    @staticmethod
    def get_csv_files_paths():
        csv_dir_path = DirHandler.get_temp_csv_dir_path()
        files_names = DirHandler.get_files_names(csv_dir_path, extension='csv')
        return list(map(lambda x: csv_dir_path + '/' + x + '.csv', files_names))

    @staticmethod
    def get_periods_dir():
        return f'{DirHandler.__get_split_data_root_path()}/{cfg.PERIOD_ROOT_DIR_NAME}'

    @staticmethod
    def get_periods_file_names():
        path = DirHandler.get_periods_dir()
        file_names = os.listdir(path)
        file_names.sort()
        return list(map(lambda file_name: f'{path}/{file_name}', file_names))


class BatchSplitter:
    lines_num_each_file = 100000
    max_num_of_files = 20
    start_line_idx = 13_000_001

    @staticmethod
    def split_to_equal_files():
        logging.info("Splitting .fasta files")
        lines_num = int(BatchSplitter.__get_num_of_lines())
        max_num_iters = BatchSplitter.__get_iterations_num(lines_num)
        start_line_idx = BatchSplitter.start_line_idx
        left_lines = lines_num - (start_line_idx - 1)
        end_line_idx = start_line_idx + min(BatchSplitter.lines_num_each_file, left_lines)
        filepath_with_name_core = DirHandler.get_temp_fasta_dir_path() + '/' + cfg.FILES_NAME_CORE
        raw_data_filepath = cfg.DATA_PARENT_PATH + '/' + cfg.DATA_RAW_FILE_NAME

        for i in range(max_num_iters):
            logging.info(f'iteration {i+1} out of {max_num_iters}')
            logging.info(f'lines {start_line_idx} to {end_line_idx}')
            copy_command = ['cat', raw_data_filepath, '|', 'sed', '-n',
                            f'{start_line_idx},{end_line_idx}p', '>>',
                            f"{filepath_with_name_core}-{start_line_idx}-{end_line_idx}.fasta"]
            os.system(' '.join(copy_command))
            start_line_idx += BatchSplitter.lines_num_each_file
            left_lines = lines_num - (start_line_idx - 1)
            end_line_idx = start_line_idx + min(BatchSplitter.lines_num_each_file, left_lines)
        logging.info(f'Finished at {end_line_idx} line')

    @staticmethod
    def __get_iterations_num(lines_num):
        max_num_of_batches = (lines_num - BatchSplitter.start_line_idx + 1) // BatchSplitter.lines_num_each_file
        return min(BatchSplitter.max_num_of_files, max_num_of_batches + 1)

    @staticmethod
    def __get_num_of_lines():
        raw_data_filepath = cfg.DATA_PARENT_PATH + '/' + cfg.DATA_RAW_FILE_NAME
        num_lines_command = f'wc -l {raw_data_filepath}'
        output = subprocess.check_output(num_lines_command, shell=True, encoding='UTF-8')
        output = output.split()
        return output[0]


class CsvTransformer:
    @staticmethod
    def transform_files():
        logging.info('Transforming .fasta to .csv')
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
    max_len = 1280

    @staticmethod
    def clean():
        logging.info('Cleaning csv files')
        files_paths = DirHandler.get_csv_files_paths()
        counter = 0
        files_num = len(files_paths)
        for file in files_paths:
            counter += 1
            logging.info(f'file {counter} out of {files_num}')
            df = pd.read_csv(file)
            df = BatchCleaner.__remove_ambiguous(df)
            df = BatchCleaner.__remove_wrong_len(df)
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
    def __remove_wrong_len(df):
        end_sign = '*'
        df = df[df['sequence'].str.endswith(end_sign)]
        df = df[(df['sequence'].str.len() >= BatchCleaner.min_len) & (df['sequence'].str.len() <= BatchCleaner.max_len)]
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

    @staticmethod
    def remove_duplicates_periods():
        logging.info('Removing duplicated periods')
        path = DirHandler.get_periods_dir()
        file_names = os.listdir(path)
        file_names.sort()
        files = list(map(lambda file_name: f'{path}/{file_name}', file_names))
        for file in files:
            df = pd.read_csv(file)
            df.drop_duplicates(subset=['isolate_name'], inplace=True)
            df.to_csv(file, index=False)
            logging.info(f'removed duplicates from {file.split("/")[-1]}')

    @staticmethod
    def remove_empty_periods_dir():
        logging.info('Removing empty periods dirs')
        files = DirHandler.get_periods_file_names()
        for file in files:
            df = pd.read_csv(file)
            rows_num = df.shape[0]
            BatchCleaner.__remove_file_if_empty(file, rows_num)

    @staticmethod
    def __remove_file_if_empty(file, rows_num):
        if rows_num == 0:
            os.remove(file)
            logging.info(f'removed: {file.split("/")[-1]}')


class PeriodSorter:
    periods_in_year_beginning = {
        'quarter': [['01-01', '03-31'], ['04-01', '06-31'], ['07-01', '09-30'], ['10-01', '12-31']],
        'month': [['01-01', '01-31'], ['02-01', '02-28'], ['03-01', '03-31'], ['04-01', '04-30'],
                  ['05-01', '05-31'], ['06-01', '06-30'], ['07-01', '07-31'], ['08-01', '08-31'],
                  ['09-01', '09-30'], ['10-01', '10-31'], ['11-01', '11-30'], ['12-01', '12-31']]}

    @staticmethod
    def divide():
        logging.info('Dividing among periods')
        logging.info(f'Dividing strategy: {cfg.DIVISION_TECHNIQUE}')
        files_paths = DirHandler.get_csv_files_paths()
        counter = 0
        files_num = len(files_paths)
        for file in files_paths:
            counter += 1
            logging.info(f'file {counter} out of {files_num}')
            df = pd.read_csv(file)
            df = PeriodSorter.__sort(df)
            PeriodSorter.__divide(df)
            df.to_csv(file, index=False)

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
        types[cfg.DIVISION_TECHNIQUE](df)

    @staticmethod
    def __divide_by_year(df):
        root_path = DirHandler.get_periods_dir()
        first_year = PeriodSorter.__get_first_year(df)
        last_year = PeriodSorter.__get_last_year(df)
        years = range(first_year, last_year + 1)
        for year in years:
            start = f'{year}-01-01'
            end = f'{year}-12-31'
            year_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            output_path = f'{root_path}/{year}.csv'
            header_flag = PeriodSorter.__should_header_be_added(output_path)
            year_df.to_csv(output_path, index=False, mode='a', header=header_flag)

    @staticmethod
    def __divide_by_quarter(df):
        root_path = DirHandler.get_periods_dir()
        first_year = PeriodSorter.__get_first_year(df)
        last_year = PeriodSorter.__get_last_year(df)
        years = range(first_year, last_year + 1)
        for year in years:
            count = 1
            for quarter in PeriodSorter.periods_in_year_beginning['quarter']:
                start = f'{year}-{quarter[0]}'
                end = f'{year}-{quarter[1]}'
                year_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
                output_path = f'{root_path}/{year}-q{str(count)}.csv'
                header_flag = PeriodSorter.__should_header_be_added(output_path)
                year_df.to_csv(output_path, index=False, mode='a', header=header_flag)
                count += 3

    @staticmethod
    def __divide_by_month(df):
        root_path = DirHandler.get_periods_dir()
        first_year = PeriodSorter.__get_first_year(df)
        last_year = PeriodSorter.__get_last_year(df)
        years = range(first_year, last_year + 1)
        for year in years:
            count = 1
            for month in PeriodSorter.periods_in_year_beginning['month']:
                start = f'{year}-{month[0]}'
                end = f'{year}-{month[1]}'
                year_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
                output_path = f'{root_path}/{year}-{str(count)}.csv'
                header_flag = PeriodSorter.__should_header_be_added(output_path)
                year_df.to_csv(output_path, index=False, mode='a', header=header_flag)
                count += 1

    @staticmethod
    def __get_first_year(df):
        first_row = df.head(1)
        first_date = pd.DataFrame(first_row['timestamp'].dt.year)
        first_date = first_date.iloc[0]['timestamp']
        return first_date.astype(int)

    @staticmethod
    def __get_last_year(df):
        last_row = df.tail(1)
        last_date = pd.DataFrame(last_row['timestamp'].dt.year)
        last_date = last_date.iloc[0]['timestamp']
        return last_date.astype(int)

    @staticmethod
    def __should_header_be_added(output_path):
        return not os.path.exists(output_path)


def prepare_files():
    DirHandler.create_dirs()
    BatchSplitter.split_to_equal_files()
    CsvTransformer.transform_files()
    DirHandler.delete_temp_fasta()
    BatchCleaner.clean()
    PeriodSorter.divide()
    DirHandler.delete_temp_csv()
    BatchCleaner.remove_duplicates_periods()
    BatchCleaner.remove_empty_periods_dir()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # TODO: think about sequences that are not exactly 1273 long
    #       think about clays of sequnces (where there is some info about the clay)
    prepare_files()
    logging.info("Done")

