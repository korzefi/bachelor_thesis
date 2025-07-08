import os
import pandas as pd
import logging
import argparse
from natsort import natsorted

def compute_centroids(input_dir, output_csv):
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    files = natsorted(files)
    centroids = []
    for file in files:
        input_path = os.path.join(input_dir, file)
        df = pd.read_csv(input_path)
        if 'cluster' not in df.columns:
            logging.warning(f'Skipping {file}: no cluster column')
            continue
        period = file.replace('.csv', '')
        for cluster_num in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_num].drop('cluster', axis=1)
            centroid = cluster_df.mean().values
            centroids.append([period, cluster_num] + centroid.tolist())
    if centroids:
        columns = ['period', 'cluster'] + [f'd{i+1}' for i in range(len(centroids[0])-2)]
        centroids_df = pd.DataFrame(centroids, columns=columns)
        centroids_df.to_csv(output_csv, index=False)
        logging.info(f'Centroids saved to {output_csv}')
    else:
        logging.warning('No centroids computed.')

def main():
    parser = argparse.ArgumentParser(description="Compute centroids for all clustered period CSVs.")
    parser.add_argument('--input_dir', required=True, help='Directory with clustered period CSVs')
    parser.add_argument('--output_csv', required=True, help='Path to save centroids CSV')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    compute_centroids(args.input_dir, args.output_csv)

if __name__ == '__main__':
    main()

