import pandas as pd
import numpy as np
import logging
import argparse
from natsort import natsort_keygen

def link_clusters(centroids_csv, output_csv):
    df = pd.read_csv(centroids_csv)
    # Sort by period and cluster
    df.sort_values(by=['period', 'cluster'], key=natsort_keygen(), inplace=True)
    
    periods = df['period'].unique()
    if len(periods) < 2:
        raise ValueError('Not enough periods to link clusters')
    
    cols = [c for c in df.columns if c.startswith('d')]
    
    # Get bidirectional links
    forward_links = {}
    backward_links = {}
    
    for i in range(len(periods) - 1):
        current_period = periods[i]
        next_period = periods[i + 1]
        forward_links.update(link_clusters_forward(df, current_period, next_period, cols))
        backward_links.update(link_clusters_backward(df, current_period, next_period, cols))
    
    # Combine forward and backward links
    combined_links = combine_links(forward_links, backward_links)
    
    # Get cluster values from combined links
    cluster_values = get_cluster_values(df['cluster'], combined_links, len(df.index))
    
    # Add next_cluster column
    df['next_cluster'] = cluster_values
    df.to_csv(output_csv, index=False)
    logging.info(f'Linked clusters saved to {output_csv}')

def link_clusters_forward(df, current_period, next_period, cols):
    """Link clusters from current period to next period (forward direction)"""
    links = {}
    current_df = df[df['period'] == current_period]
    next_df = df[df['period'] == next_period]
    
    for idx1, row1 in current_df.iterrows():
        min_dist = float('inf')
        min_idx = None
        for idx2, row2 in next_df.iterrows():
            dist = get_euclidean_distance(row1, row2, cols)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx2
        links[idx1] = min_idx
    return links

def link_clusters_backward(df, current_period, next_period, cols):
    """Link clusters from next period to current period (backward direction)"""
    links = {}
    current_df = df[df['period'] == current_period]
    next_df = df[df['period'] == next_period]
    
    for idx2, row2 in next_df.iterrows():
        min_dist = float('inf')
        min_idx = None
        for idx1, row1 in current_df.iterrows():
            dist = get_euclidean_distance(row1, row2, cols)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx1
        links[idx2] = min_idx
    return links

def get_euclidean_distance(row1, row2, cols):
    """Calculate Euclidean distance between two centroid vectors"""
    vec1 = np.array(row1[cols], dtype=float)
    vec2 = np.array(row2[cols], dtype=float)
    return np.linalg.norm(vec1 - vec2)

def combine_links(forward_links, backward_links):
    """Combine forward and backward links"""
    # Transform forward links to lists
    forward_links = {k: [v] for k, v in forward_links.items()}
    
    # Reverse backward links
    reversed_backward = {}
    for key, value in backward_links.items():
        if value in reversed_backward:
            reversed_backward[value].append(key)
        else:
            reversed_backward[value] = [key]
    
    # Combine links
    combined = {}
    combined.update(forward_links)
    
    for k, v in reversed_backward.items():
        if k in combined:
            combined[k].extend(v)
        else:
            combined[k] = v
    
    # Remove duplicates and sort
    combined = {k: sorted(list(set(v))) for k, v in combined.items()}
    return combined

def get_cluster_values(clusters_data, combined_links, total_rows):
    """Convert combined links to cluster values in string format"""
    # Transform to clusters
    clusters = {}
    for idx, idx_links in combined_links.items():
        clusters[idx] = [clusters_data.iloc[idx_link] for idx_link in idx_links]
    
    # Fill missing clusters with empty string
    for i in range(total_rows):
        if i not in clusters:
            clusters[i] = []
    
    # Convert to string format (e.g., '0-2-3' or '' for empty)
    cluster_values = []
    for i in range(total_rows):
        if clusters[i]:
            cluster_str = '-'.join(map(str, clusters[i]))
        else:
            cluster_str = ''
        cluster_values.append(cluster_str)
    
    return cluster_values

def main():
    parser = argparse.ArgumentParser(description="Link clusters between consecutive periods based on centroid similarity.")
    parser.add_argument('--centroids_csv', required=True, help='Path to centroids CSV')
    parser.add_argument('--output_csv', required=True, help='Path to save linked clusters CSV')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    link_clusters(args.centroids_csv, args.output_csv)

if __name__ == '__main__':
    main()

