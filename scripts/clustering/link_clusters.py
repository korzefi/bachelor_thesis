import pandas as pd
import numpy as np
import logging
import argparse

def link_clusters(centroids_csv, output_csv):
    df = pd.read_csv(centroids_csv)
    periods = df['period'].unique()
    cols = [c for c in df.columns if c.startswith('d')]
    next_clusters = []
    for i, row in df.iterrows():
        period = row['period']
        cluster = row['cluster']
        # Find next period
        try:
            next_period = periods[list(periods).index(period) + 1]
        except IndexError:
            next_clusters.append(np.nan)
            continue
        current_vec = row[cols].values.astype(float)
        next_df = df[df['period'] == next_period]
        if next_df.empty:
            next_clusters.append(np.nan)
            continue
        # Find closest centroid in next period
        dists = next_df[cols].apply(lambda x: np.linalg.norm(current_vec - x.values.astype(float)), axis=1)
        min_idx = dists.idxmin()
        next_cluster = next_df.loc[min_idx, 'cluster']
        next_clusters.append(next_cluster)
    df['next_cluster'] = next_clusters
    df.to_csv(output_csv, index=False)
    logging.info(f'Linked clusters saved to {output_csv}')

def main():
    parser = argparse.ArgumentParser(description="Link clusters between consecutive periods based on centroid similarity.")
    parser.add_argument('--centroids_csv', required=True, help='Path to centroids CSV')
    parser.add_argument('--output_csv', required=True, help='Path to save linked clusters CSV')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    link_clusters(args.centroids_csv, args.output_csv)

if __name__ == '__main__':
    main()

