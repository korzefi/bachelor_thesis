import os
import pandas as pd
import logging
import argparse
from natsort import natsorted

def clean_csv_dir(csv_dir, expected_len=1273, error_margin=0.5):
    # Compute min_len and max_len from expected_len and error_margin
    margin = int(expected_len * (error_margin / 100))
    min_len = expected_len - margin
    max_len = expected_len + margin
    
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    files = natsorted(files)
    for file in files:
        path = os.path.join(csv_dir, file)
        df = pd.read_csv(path)
        df = remove_ambiguous(df)
        df = remove_wrong_len(df, min_len, max_len)
        df = filter_description(df)
        df = remove_duplicates(df)
        df.to_csv(path, index=False)
        logging.info(f'Cleaned {file}')
    logging.info(f'Finished cleaning all CSV files in {csv_dir}')

def remove_ambiguous(df):
    ambiguous_aminos = ['B', 'J', 'Z', 'X', '-']
    df = df[~df['sequence'].str.contains('|'.join(ambiguous_aminos))]
    df.reset_index(drop=True, inplace=True)
    return df

def remove_wrong_len(df, min_len, max_len):
    end_sign = '*'
    df = df[df['sequence'].str.endswith(end_sign)]
    df = df[(df['sequence'].str.len() >= min_len) & (df['sequence'].str.len() <= max_len)]
    return df

def filter_description(df):
    splitted_desc = df['description'].str.split(pat='|', expand=True, n=3)
    splitted_desc.columns = ['gene', 'isolate_name', 'timestamp', 'rest']
    splitted_desc.drop(columns=['gene', 'rest'], inplace=True)
    splitted_desc = adapt_day_format(splitted_desc)
    df.drop(columns='description', inplace=True)
    df = pd.concat([splitted_desc, df], axis=1, join='inner')
    return df

def adapt_day_format(df):
    days = pd.to_numeric(df['timestamp'].str[-2:]) + 1
    df['timestamp'] = df['timestamp'].str[:-2] + days.astype(str)
    return df

def remove_duplicates(df):
    df.drop_duplicates(subset=['isolate_name'], inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Clean all CSV files in a directory.")
    parser.add_argument('--csv_dir', required=True, help='Directory containing CSV files to clean')
    parser.add_argument('--expected_len', type=int, default=1273, help='Expected sequence length')
    parser.add_argument('--error_margin', type=float, default=0.5, help='Error margin percentage')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    clean_csv_dir(args.csv_dir, args.expected_len, args.error_margin)

if __name__ == '__main__':
    main()

