import os
import pandas as pd
import logging
import argparse
from natsort import natsorted
from sklearn.cluster import KMeans

def cluster_sequences(input_dir, output_dir, n_clusters=8):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    files = natsorted(files)
    for file in files:
        input_path = os.path.join(input_dir, file)
        df = pd.read_csv(input_path)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(df)
        df_out = df.copy()
        df_out.insert(0, 'cluster', labels)
        out_path = os.path.join(output_dir, file)
        df_out.to_csv(out_path, index=False)
        logging.info(f'Clustered {file} into {n_clusters} clusters, saved to {out_path}')
    logging.info(f'Finished clustering all vectorized files in {input_dir} to {output_dir}')

def main():
    parser = argparse.ArgumentParser(description="Cluster all vectorized period CSVs using KMeans.")
    parser.add_argument('--input_dir', required=True, help='Directory with vectorized period CSVs')
    parser.add_argument('--output_dir', required=True, help='Directory to save clustered CSVs')
    parser.add_argument('--n_clusters', type=int, default=8, help='Number of clusters for KMeans')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cluster_sequences(args.input_dir, args.output_dir, args.n_clusters)

if __name__ == '__main__':
    main()

