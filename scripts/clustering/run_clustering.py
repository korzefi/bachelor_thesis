import argparse
import subprocess
import logging
import os
import yaml

def run_pipeline(args):
    logging.basicConfig(level=logging.INFO)
    # 1. Vectorize sequences
    vector_dir = args.vector_dir
    logging.info('Step 1: Vectorizing sequences...')
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'vectorize_sequences.py'),
        '--input_dir', args.input_dir,
        '--output_dir', vector_dir,
        '--protvec_path', args.protvec_path
    ], check=True)
    # 2. Cluster sequences
    cluster_dir = args.cluster_dir
    logging.info('Step 2: Clustering sequences...')
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'cluster_sequences.py'),
        '--input_dir', vector_dir,
        '--output_dir', cluster_dir,
        '--n_clusters', str(args.n_clusters)
    ], check=True)
    # 3. Compute centroids
    centroids_csv = args.centroids_csv
    logging.info('Step 3: Computing centroids...')
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'compute_centroids.py'),
        '--input_dir', cluster_dir,
        '--output_csv', centroids_csv
    ], check=True)
    # 4. Link clusters
    linked_csv = args.linked_csv
    logging.info('Step 4: Linking clusters...')
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'link_clusters.py'),
        '--centroids_csv', centroids_csv,
        '--output_csv', linked_csv
    ], check=True)
    logging.info('Clustering pipeline completed successfully.')

def main():
    parser = argparse.ArgumentParser(description='Run the full clustering pipeline.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (optional)')
    parser.add_argument('--input_dir', default=None)
    parser.add_argument('--vector_dir', default=None)
    parser.add_argument('--cluster_dir', default=None)
    parser.add_argument('--centroids_csv', default=None)
    parser.add_argument('--linked_csv', default=None)
    parser.add_argument('--protvec_path', default=None)
    parser.add_argument('--n_clusters', type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    def get_param(key, default=None):
        return getattr(args, key) if getattr(args, key) not in [None, 'None'] else config.get(key, default)

    class Args:
        pass
    merged_args = Args()
    merged_args.input_dir = get_param('input_dir')
    merged_args.vector_dir = get_param('vector_dir')
    merged_args.cluster_dir = get_param('cluster_dir')
    merged_args.centroids_csv = get_param('centroids_csv')
    merged_args.linked_csv = get_param('linked_csv')
    merged_args.protvec_path = get_param('protvec_path')
    merged_args.n_clusters = int(get_param('n_clusters', 8))

    run_pipeline(merged_args)

if __name__ == '__main__':
    main()
