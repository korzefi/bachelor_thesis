import os
import pandas as pd
import logging
import argparse
from natsort import natsorted

def fasta_to_csv(fasta_dir, csv_dir):
    os.makedirs(csv_dir, exist_ok=True)
    files = [f for f in os.listdir(fasta_dir) if f.endswith('.fasta')]
    files = natsorted(files)
    for name in files:
        read_file = pd.read_fwf(os.path.join(fasta_dir, name), header=None)
        description_df = read_file.iloc[::2, :]
        description_df.columns = ['description']
        description_df.reset_index(drop=True, inplace=True)
        sequence_df = read_file.iloc[1::2, :]
        sequence_df.columns = ['sequence']
        sequence_df.reset_index(drop=True, inplace=True)
        result = pd.concat([description_df, sequence_df], axis=1, join='inner')
        out_csv = os.path.join(csv_dir, name.replace('.fasta', '.csv'))
        result.to_csv(out_csv, index=False)
        logging.info(f'Converted {name} to {out_csv}')
    logging.info(f'Finished converting all FASTA files in {fasta_dir} to CSV in {csv_dir}')

def main():
    parser = argparse.ArgumentParser(description="Convert all FASTA files in a directory to CSV format.")
    parser.add_argument('--fasta_dir', required=True, help='Directory containing FASTA files')
    parser.add_argument('--csv_dir', required=True, help='Directory to save CSV files')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    fasta_to_csv(args.fasta_dir, args.csv_dir)

if __name__ == '__main__':
    main()
