#!/usr/bin/env python3
"""
Clustering Pipeline Module

This module handles the creation of clusters from sequence data by:
1. Converting sequences to triplets (3-grams of amino acids)
2. Transforming triplets to vectors using ProtVec embeddings
3. Creating clusters using K-means clustering
4. Saving cluster centroids and labels

Refactored from: scripts/preprocessing/clustering.py, ClusterCentroidsDataCreator.py
"""

import os
import logging
import multiprocessing
import pandas as pd
from pathlib import Path
from natsort import natsorted
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import scripts.utils as utils


class ClusteringPipeline:
    """Main class that orchestrates the clustering process."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration dictionary."""
        self.config = config
        self.data_config = config['data']
        self.cluster_config = config['cluster']
        
        # Setup paths
        self.periods_unique_dir = self.data_config['periods_unique_dir']
        self.vector_temp_dir = self.data_config['vector_temp_dir']
        self.prot_vec_path = self.data_config['prot_vec_file']
        self.centroids_file = self.data_config['centroids_file']
        
        # Create directories
        self._create_directories()
        
        # Load ProtVec embeddings
        self.prot_vec_df = self._load_prot_vec_embeddings()
    
    def _create_directories(self) -> None:
        """Create necessary directory structure."""
        utils.create_dir(self.vector_temp_dir)
        utils.create_dir(os.path.dirname(self.centroids_file))
    
    def _load_prot_vec_embeddings(self) -> pd.DataFrame:
        """Load ProtVec 100-dimensional embeddings."""
        logging.info(f"Loading ProtVec embeddings from: {self.prot_vec_path}")
        return pd.read_csv(self.prot_vec_path, sep='\t')
    
    def run(self) -> None:
        """Execute the complete clustering pipeline."""
        logging.info("Starting clustering pipeline...")
        
        try:
            # Step 1: Get period files to process
            period_files = self._get_period_files()
            logging.info(f"Found {len(period_files)} period files to process")
            
            # Step 2: Transform sequences to vectors
            if self.cluster_config['use_multiprocessing']:
                self._transform_sequences_to_vectors_parallel(period_files)
            else:
                self._transform_sequences_to_vectors_sequential(period_files)
            
            # Step 3: Create clusters for each period
            self._create_clusters_for_periods()
            
            logging.info("Clustering pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Clustering pipeline failed: {e}")
            raise
    
    def _get_period_files(self) -> List[str]:
        """Get list of period files to process."""
        if not os.path.exists(self.periods_unique_dir):
            raise FileNotFoundError(f"Periods directory not found: {self.periods_unique_dir}")
        
        files = [f for f in os.listdir(self.periods_unique_dir) if f.endswith('.csv')]
        files = natsorted(files)
        
        # Check if auto k-selection is enabled
        auto_k_config = self.cluster_config.get('auto_k_selection', {})
        auto_k_enabled = auto_k_config.get('enabled', False)
        
        if auto_k_enabled:
            # Process all CSV files when auto k-selection is enabled
            return files
        else:
            # Filter files based on clusters_per_period configuration (legacy mode)
            clusters_per_period = self.cluster_config.get('clusters_per_period', {})
            
            if not clusters_per_period:
                logging.warning("No clusters_per_period configuration found, processing all files")
                return files
            
            filtered_files = [f for f in files if f in clusters_per_period]
            
            if not filtered_files:
                logging.warning("No files match the clusters_per_period configuration")
                return files  # Return all files if none match config
            
            return filtered_files
    
    def _transform_sequences_to_vectors_sequential(self, period_files: List[str]) -> None:
        """Transform sequences to vectors sequentially."""
        logging.info("Transforming sequences to vectors (sequential)...")
        
        for i, period_file in enumerate(period_files, 1):
            logging.info(f"Processing file {i}/{len(period_files)}: {period_file}")
            self._transform_single_file(period_file)
    
    def _transform_sequences_to_vectors_parallel(self, period_files: List[str]) -> None:
        """Transform sequences to vectors using multiprocessing."""
        logging.info("Transforming sequences to vectors (parallel)...")
        
        # Determine number of processes
        processes = self.cluster_config.get('processes')
        if processes is None:
            processes = multiprocessing.cpu_count()
        
        logging.info(f"Using {processes} processes for parallel transformation")
        
        # Create process pool
        with multiprocessing.Pool(processes=processes) as pool:
            pool.map(self._transform_single_file, period_files)
    
    def _transform_single_file(self, period_file: str) -> None:
        """Transform sequences in a single file to vectors."""
        logging.info(f"Transforming sequences for file: {period_file}")
        
        input_path = f"{self.periods_unique_dir}/{period_file}"
        output_path = f"{self.vector_temp_dir}/{period_file}"
        
        # Load sequences
        df = pd.read_csv(input_path)
        sequences = df['sequence'].tolist()
        
        # Convert sequences to triplets
        triplets_list = self._create_triplets_from_sequences(sequences)
        
        # Transform triplets to vectors
        vectors_df = self._transform_triplets_to_vectors(triplets_list)
        
        # Save vectors
        vectors_df.to_csv(output_path, index=False)
        
        logging.info(f"Completed transformation for {period_file}: {len(vectors_df)} vectors")
    
    def _create_triplets_from_sequences(self, sequences: List[str]) -> List[List[str]]:
        """Convert sequences to 3-grams (triplets) of amino acids."""
        triplets_list = []
        
        for sequence in sequences:
            # Remove stop codon if present
            if sequence.endswith('*'):
                sequence = sequence[:-1]
            
            # Create triplets
            seq_len = len(sequence)
            if seq_len >= 3:
                triplets = [sequence[i:i+3] for i in range(seq_len - 2)]
                triplets_list.append(triplets)
            else:
                # Handle very short sequences
                triplets_list.append([sequence])
        
        return triplets_list
    
    def _transform_triplets_to_vectors(self, triplets_list: List[List[str]]) -> pd.DataFrame:
        """Transform triplets to vectors using ProtVec embeddings."""
        # Create template for results (100-dimensional vectors)
        vector_columns = [f'd{i}' for i in range(1, 101)]
        result_df = pd.DataFrame(columns=vector_columns)
        
        for i, triplets in enumerate(triplets_list):
            logging.debug(f"Processing sequence {i+1}/{len(triplets_list)}")
            
            # Initialize sequence vector with zeros
            seq_vector = pd.Series([0.0] * 100, index=vector_columns)
            
            # Sum up vectors for all triplets in the sequence
            for triplet in triplets:
                # Find triplet in ProtVec embeddings
                triplet_row = self.prot_vec_df[self.prot_vec_df['words'] == triplet]
                
                if not triplet_row.empty:
                    # Add the embedding vector to the sequence vector
                    triplet_vector = triplet_row.iloc[0, 1:101]  # Skip 'words' column
                    seq_vector += triplet_vector.values
            
            # Round to 8 decimal places
            seq_vector = seq_vector.round(8)
            
            # Add to results
            result_df.loc[len(result_df)] = seq_vector
        
        return result_df
    
    def _find_optimal_k(self, vectors_df: pd.DataFrame) -> int:
        """Find optimal number of clusters using Silhouette Score."""
        # Get k range from config or use defaults
        auto_k_config = self.cluster_config.get('auto_k_selection', {})
        min_k = auto_k_config.get('min_k', 2)
        max_k = auto_k_config.get('max_k', 15)
        
        # Ensure we don't try more clusters than we have data points
        n_samples = len(vectors_df)
        max_k = min(max_k, n_samples - 1)
        
        if min_k >= n_samples:
            logging.warning(f"Not enough samples ({n_samples}) for clustering. Using k=1")
            return 1
        
        best_k = min_k
        best_score = -1
        
        logging.debug(f"Finding optimal k in range [{min_k}, {max_k}] for {n_samples} samples")
        
        vectors_array = vectors_df.to_numpy()
        
        for k in range(min_k, max_k + 1):
            # Fit K-means
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
            labels = kmeans.fit_predict(vectors_array)
            
            # Calculate silhouette score
            score = silhouette_score(vectors_array, labels)
            
            logging.debug(f"k={k}: silhouette_score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        logging.info(f"Optimal k={best_k} with silhouette_score={best_score:.4f}")
        return best_k
    
    def _create_clusters_for_periods(self) -> None:
        """Create clusters for each period using K-means."""
        logging.info("Creating clusters for each period...")
        
        # Initialize centroids file
        self._initialize_centroids_file()
        
        # Check if auto k-selection is enabled
        auto_k_config = self.cluster_config.get('auto_k_selection', {})
        auto_k_enabled = auto_k_config.get('enabled', False)
        
        if auto_k_enabled:
            # Use automatic k-finding for all vector files
            vector_files = [f for f in os.listdir(self.vector_temp_dir) if f.endswith('.csv')]
            vector_files = natsorted(vector_files)
            
            for period_file in vector_files:
                logging.info(f"Processing {period_file} with automatic k-selection")
                
                vector_file_path = f"{self.vector_temp_dir}/{period_file}"
                
                # Load vectors
                vectors_df = pd.read_csv(vector_file_path)
                
                if vectors_df.empty:
                    logging.warning(f"Empty vector file: {period_file}")
                    continue
                
                # Find optimal k
                n_clusters = self._find_optimal_k(vectors_df)
                logging.info(f"Using {n_clusters} clusters for {period_file}")
                
                # Create clusters
                cluster_results = self._create_kmeans_clusters(vectors_df, n_clusters)
                
                # Save cluster labels to original period file
                self._save_cluster_labels(period_file, cluster_results['labels'])
                
                # Save centroids
                self._save_centroids(period_file, cluster_results['centroids'])
        else:
            # Use manual clusters_per_period configuration (legacy mode)
            clusters_per_period = self.cluster_config.get('clusters_per_period', {})
            
            if not clusters_per_period:
                raise ValueError("Either enable auto_k_selection or provide clusters_per_period configuration")
            
            for period_file, n_clusters in clusters_per_period.items():
                logging.info(f"Creating {n_clusters} clusters for {period_file}")
                
                vector_file_path = f"{self.vector_temp_dir}/{period_file}"
                
                if not os.path.exists(vector_file_path):
                    logging.warning(f"Vector file not found: {vector_file_path}")
                    continue
                
                # Load vectors
                vectors_df = pd.read_csv(vector_file_path)
                
                if vectors_df.empty:
                    logging.warning(f"Empty vector file: {period_file}")
                    continue
                
                # Create clusters
                cluster_results = self._create_kmeans_clusters(vectors_df, n_clusters)
                
                # Save cluster labels to original period file
                self._save_cluster_labels(period_file, cluster_results['labels'])
                
                # Save centroids
                self._save_centroids(period_file, cluster_results['centroids'])
        
        logging.info("Cluster creation completed for all periods")
    
    def _create_kmeans_clusters(self, vectors_df: pd.DataFrame, n_clusters: int) -> Dict:
        """Create K-means clusters from vector data."""
        # Use K-means with good default parameters
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,  # Number of random initializations
            max_iter=300,
            random_state=42  # For reproducibility
        )
        
        # Fit the model
        vectors_array = vectors_df.to_numpy()
        kmeans.fit(vectors_array)
        
        return {
            'labels': kmeans.labels_,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
    
    def _save_cluster_labels(self, period_file: str, labels: List[int]) -> None:
        """Save cluster labels to the original period file."""
        original_file_path = f"{self.periods_unique_dir}/{period_file}"
        
        # Load original data
        df = pd.read_csv(original_file_path)
        
        # Remove existing cluster column if it exists
        if 'cluster' in df.columns:
            df.drop('cluster', axis=1, inplace=True)
        
        # Add cluster labels
        df.insert(0, 'cluster', labels)
        
        # Save back to file
        df.to_csv(original_file_path, index=False)
        
        logging.debug(f"Added cluster labels to {period_file}")
    
    def _initialize_centroids_file(self) -> None:
        """Initialize the centroids CSV file with proper headers."""
        if os.path.exists(self.centroids_file):
            logging.debug(f"Centroids file already exists: {self.centroids_file}")
            return
        
        # Create column headers
        columns = ['period', 'cluster'] + [f'd{i}' for i in range(1, 101)]
        
        # Create empty DataFrame with headers
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.centroids_file, index=False)
        
        logging.info(f"Initialized centroids file: {self.centroids_file}")
    
    def _save_centroids(self, period_file: str, centroids) -> None:
        """Save cluster centroids to the centroids file."""
        period_name = period_file[:-4]  # Remove .csv extension
        
        # Load existing centroids file
        centroids_df = pd.read_csv(self.centroids_file)
        
        # Create rows for each centroid
        for cluster_num, centroid in enumerate(centroids):
            row_data = [period_name, cluster_num] + centroid.tolist()
            
            # Create DataFrame for the new row
            columns = centroids_df.columns
            new_row = pd.DataFrame([row_data], columns=columns)
            
            # Append to centroids DataFrame
            centroids_df = pd.concat([centroids_df, new_row], ignore_index=True)
        
        # Save updated centroids file
        centroids_df.to_csv(self.centroids_file, index=False)
        
        logging.debug(f"Saved {len(centroids)} centroids for {period_name}")


def run(config: Dict) -> None:
    """Main entry point for the clustering pipeline."""
    pipeline = ClusteringPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    # For testing purposes
    import sys
    sys.path.append('..')
    from scripts.config import load_config
    
    utils.setup_logger(verbose=True)
    config = load_config('../../configs/sars_cov_2_default.yaml')
    run(config)