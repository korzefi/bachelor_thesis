import os
import pandas as pd
import logging
import argparse
from natsort import natsorted

def distribute_periods(csv_dir, output_dir, division='month'):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    files = natsorted(files)
    for file in files:
        path = os.path.join(csv_dir, file)
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d', errors='coerce')
        df.sort_values(by='timestamp', inplace=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        if division == 'year':
            divide_by_year(df, output_dir)
        elif division == 'quarter':
            divide_by_quarter(df, output_dir)
        elif division == 'month':
            divide_by_month(df, output_dir)
        else:
            raise ValueError(f'Unknown division: {division}')
        logging.info(f'Distributed {file} into periods in {output_dir}')
    logging.info(f'Finished distributing all CSV files in {csv_dir} to {output_dir}')

def divide_by_year(df, output_dir):
    years = df['timestamp'].dt.year.unique()
    for year in years:
        year_df = df[df['timestamp'].dt.year == year]
        out_path = os.path.join(output_dir, f'{year}.csv')
        year_df.to_csv(out_path, index=False, mode='a', header=not os.path.exists(out_path))

def divide_by_quarter(df, output_dir):
    for year in df['timestamp'].dt.year.unique():
        for quarter in range(1, 5):
            quarter_df = df[(df['timestamp'].dt.year == year) & (df['timestamp'].dt.quarter == quarter)]
            if not quarter_df.empty:
                out_path = os.path.join(output_dir, f'{year}-Q{quarter}.csv')
                quarter_df.to_csv(out_path, index=False, mode='a', header=not os.path.exists(out_path))

def divide_by_month(df, output_dir):
    for year in df['timestamp'].dt.year.unique():
        for month in range(1, 13):
            month_df = df[(df['timestamp'].dt.year == year) & (df['timestamp'].dt.month == month)]
            if not month_df.empty:
                out_path = os.path.join(output_dir, f'{year}-{month}.csv')
                month_df.to_csv(out_path, index=False, mode='a', header=not os.path.exists(out_path))

def main():
    parser = argparse.ArgumentParser(description="Distribute cleaned CSV files into periods (month/quarter/year).")
    parser.add_argument('--csv_dir', required=True, help='Directory containing cleaned CSV files')
    parser.add_argument('--output_dir', required=True, help='Directory to save periodized CSV files')
    parser.add_argument('--division', choices=['year', 'quarter', 'month'], default='month', help='Period division type')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    distribute_periods(args.csv_dir, args.output_dir, args.division)

if __name__ == '__main__':
    main()

