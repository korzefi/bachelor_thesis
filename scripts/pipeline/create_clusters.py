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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np

import scripts.utils as utils
from scripts.utils import BatchProcessor, DataFrameChunker


class BatchVectorProcessor(BatchProcessor):
    """Batch processor for converting sequences to vectors."""
    
    def __init__(self, config: Dict, prot_vec_df: pd.DataFrame):
        memory_config = config.get('memory_optimization', {})
        batch_size = memory_config.get('clustering_batch_size', 500)
        super().__init__(config, batch_size, "BatchVectorProcessor")

        self.prot_vec_df = prot_vec_df
        self.temp_dir = None
        self.current_file = None

        # Create lookup dictionary with LRU cache
        from functools import lru_cache

        self.triplet_to_vector = {}
        for idx, row in prot_vec_df.iterrows():
            vector = row.drop('words').values.astype('float32')  # Use float32
            self.triplet_to_vector[row['words']] = vector

        # Add LRU cache for triplet sequences
        self._cached_sequence_vectors = lru_cache(maxsize=10000)(self._compute_sequence_vector)

        logging.info(f"Created vector lookup with {len(self.triplet_to_vector)} triplets")
    
    def setup_output_file(self, output_path: str):
        """Setup output file for batch results."""
        self.current_file = output_path
        self.temp_dir = f"{output_path}_temp"
        utils.create_dir(self.temp_dir)
    
    def process_batch(self, batch_sequences: List[str]) -> pd.DataFrame:
        """Process a batch of sequences to vectors."""
        # Convert sequences to triplets
        triplets_list = self._create_triplets_from_sequences(batch_sequences)
        
        # Transform triplets to vectors
        vectors_df = self._transform_triplets_to_vectors_batch(triplets_list)
        
        return vectors_df
    
    def save_batch_result(self, vectors_df: pd.DataFrame, batch_index: int) -> None:
        """Save batch result to temporary file."""
        temp_file = f"{self.temp_dir}/batch_{batch_index:04d}.csv"
        vectors_df.to_csv(temp_file, index=False)
    
    def finalize_output(self) -> None:
        """Merge all batch files into final output."""
        if not self.current_file or not self.temp_dir:
            return
        
        # Get all batch files
        batch_files = [f"{self.temp_dir}/{f}" for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
        batch_files.sort()  # Ensure correct order
        
        if batch_files:
            DataFrameChunker.merge_csv_files(batch_files, self.current_file, remove_input=True)
            os.rmdir(self.temp_dir)
            logging.info(f"Merged {len(batch_files)} batch files into {self.current_file}")
    
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
    
    def _transform_triplets_to_vectors_batch(self, triplets_list: List[List[str]]) -> pd.DataFrame:
        """Transform triplets to vectors using numpy operations."""
        batch_size = len(triplets_list)
        vector_dim = 100

        # Pre-allocate with float32 instead of float64
        result_vectors = np.zeros((batch_size, vector_dim), dtype='float32')

        # Vectorized processing
        for seq_idx, triplets in enumerate(triplets_list):
            # Use numpy array operations
            seq_vector = np.zeros(vector_dim, dtype='float32')

            for triplet in triplets:
                if triplet in self.triplet_to_vector:
                    # In-place addition
                    np.add(seq_vector, self.triplet_to_vector[triplet], out=seq_vector)

            result_vectors[seq_idx] = seq_vector

        # Create DataFrame with optimized dtypes
        vector_columns = [f'd{i}' for i in range(1, vector_dim + 1)]
        return pd.DataFrame(result_vectors, columns=vector_columns)

    def _compute_sequence_vector(self, sequence_hash: int) -> np.ndarray:
        """Cached computation of sequence vectors."""
        # This is a placeholder - would need actual sequence to compute
        # In practice, this would be used for frequently accessed sequences
        pass


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
        
        # Initialize batch processor for vector transformation
        self.batch_processor = BatchVectorProcessor(config, self.prot_vec_df)
    
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

            # TODO: if data is already embedded, don't transform it
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
        """Transform sequences in a single file to vectors using batch processing."""
        logging.info(f"Transforming sequences for file: {period_file}")
        
        input_path = f"{self.periods_unique_dir}/{period_file}"
        output_path = f"{self.vector_temp_dir}/{period_file}"
        
        # Check if streaming is enabled
        memory_config = self.config.get('memory_optimization', {})
        use_streaming = memory_config.get('use_streaming', True)
        
        if use_streaming:
            # Use batch processing for memory efficiency
            self.batch_processor.setup_output_file(output_path)
            
            # Process sequences in batches
            total_sequences = sum(1 for _ in open(input_path)) - 1  # Subtract header
            logging.info(f"Processing {total_sequences} sequences in batches")
            
            # Read and process in chunks
            self.batch_processor.process_in_batches(
                input_path, 
                description=f"vector transformation for {period_file}"
            )
            
            # Merge batch results
            self.batch_processor.finalize_output()
        else:
            # Fallback to original method for smaller files
            self._transform_single_file_legacy(period_file)
        
        # Log completion
        if os.path.exists(output_path):
            df_result = pd.read_csv(output_path)
            logging.info(f"Completed transformation for {period_file}: {len(df_result)} vectors")
    
    def _transform_single_file_legacy(self, period_file: str) -> None:
        """Legacy method for transforming sequences (for small files or fallback)."""
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
        result_df = pd.DataFrame(columns=vector_columns, dtype='float64')
        
        for i, triplets in enumerate(triplets_list):
            logging.debug(f"Processing sequence {i+1}/{len(triplets_list)}")
            
            # Initialize sequence vector with zeros
            seq_vector = pd.Series([0.0] * 100, index=vector_columns, dtype='float64')
            
            # Sum up vectors for all triplets in the sequence
            for triplet in triplets:
                # Find triplet in ProtVec embeddings
                triplet_row = self.prot_vec_df[self.prot_vec_df['words'] == triplet]
                
                if not triplet_row.empty:
                    # Add the embedding vector to the sequence vector
                    triplet_vector = triplet_row.iloc[0, 1:101]  # Skip 'words' column
                    seq_vector = seq_vector.add(triplet_vector.values, fill_value=0.0)
            
            # Round to 8 decimal places and ensure float64 dtype
            seq_vector = seq_vector.astype('float64').round(8)
            
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
        # Check if parallel processing should be used
        performance_config = self.config.get('optimization', {}).get('performance', {})
        use_multiprocessing = performance_config.get('use_multiprocessing', False)

        if use_multiprocessing:
            self._create_clusters_for_periods_parallel()
        else:
            self._create_clusters_for_periods_sequential()

    def _create_clusters_for_periods_sequential(self) -> None:
        """Create clusters for each period using K-means (sequential)."""
        logging.info("Creating clusters for each period (sequential)...")

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
                vectors_df = pd.read_csv(vector_file_path, dtype='float64')
                
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
                vectors_df = pd.read_csv(vector_file_path, dtype='float64')
                
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

    def _create_clusters_for_periods_parallel(self) -> None:
        """Create clusters using parallel processing."""
        from joblib import Parallel, delayed

        logging.info("Creating clusters for each period (parallel)...")

        # Initialize centroids file
        self._initialize_centroids_file()

        # Get vector files
        vector_files = natsorted([f for f in os.listdir(self.vector_temp_dir) if f.endswith('.csv')])

        # Determine number of jobs
        n_jobs = min(len(vector_files), os.cpu_count())

        # Process in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_single_period_clustering)(
                period_file,
                self.vector_temp_dir,
                self.cluster_config
            ) for period_file in vector_files
        )

        # Collect and save results
        for period_file, centroids, labels in results:
            if centroids is not None:
                self._save_cluster_labels(period_file, labels)
                self._save_centroids(period_file, centroids)

    @staticmethod
    def _process_single_period_clustering(period_file: str, vector_temp_dir: str,
                                         cluster_config: dict) -> tuple:
        """Process clustering for single period."""
        try:
            vector_path = f"{vector_temp_dir}/{period_file}"
            vectors_df = pd.read_csv(vector_path, dtype='float32')

            if vectors_df.empty:
                return period_file, None, None

            # Determine number of clusters
            if cluster_config.get('auto_k_selection', {}).get('enabled', False):
                n_clusters = ClusteringPipeline._find_optimal_k_static(vectors_df.values, cluster_config)
            else:
                n_clusters = cluster_config['clusters_per_period'].get(period_file, 5)

            # Use MiniBatchKMeans for speed
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(500, len(vectors_df) // 4),
                n_init=3,
                max_iter=100,
                random_state=42
            )

            labels = kmeans.fit_predict(vectors_df.values)

            return period_file, kmeans.cluster_centers_, labels

        except Exception as e:
            logging.error(f"Clustering failed for {period_file}: {e}")
            return period_file, None, None

    @staticmethod
    def _find_optimal_k_static(vectors_array, cluster_config: dict) -> int:
        """Static method for finding optimal k (for parallel processing)."""
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

        auto_k_config = cluster_config.get('auto_k_selection', {})
        min_k = auto_k_config.get('min_k', 2)
        max_k = auto_k_config.get('max_k', 15)

        # Ensure we don't try more clusters than we have data points
        n_samples = len(vectors_array)
        max_k = min(max_k, n_samples - 1)

        if min_k >= n_samples:
            logging.warning(f"Not enough samples ({n_samples}) for clustering. Using k=1")
            return 1

        best_k = min_k
        best_score = -1

        for k in range(min_k, max_k + 1):
            # Fit K-means
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
            labels = kmeans.fit_predict(vectors_array)

            # Calculate silhouette score
            score = silhouette_score(vectors_array, labels)

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def _create_kmeans_clusters(self, vectors_df: pd.DataFrame, n_clusters: int) -> Dict:
        """Create K-means clusters from vector data with memory optimization."""
        vectors_array = vectors_df.to_numpy()
        n_samples = len(vectors_array)
        
        # Get memory optimization settings
        memory_config = self.config.get('memory_optimization', {})
        clustering_batch_size = memory_config.get('clustering_batch_size', 500)
        
        # Choose clustering algorithm based on data size
        use_minibatch = n_samples > clustering_batch_size * 2
        
        if use_minibatch:
            logging.info(f"Using Mini-Batch K-means for {n_samples} samples (memory optimization)")
            
            # Use Mini-Batch K-means for large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(clustering_batch_size, n_samples // 4),
                n_init=3,  # Fewer initializations for Mini-Batch
                max_iter=100,
                random_state=42,
                max_no_improvement=10
            )
        else:
            logging.info(f"Using standard K-means for {n_samples} samples")
            
            # Use standard K-means for smaller datasets
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10,  # Number of random initializations
                max_iter=300,
                random_state=42  # For reproducibility
            )
        
        # Fit the model with memory monitoring
        initial_memory = self.batch_processor.memory_monitor.get_memory_usage_mb()
        logging.info(f"Starting clustering with {initial_memory:.1f}MB memory usage")
        
        kmeans.fit(vectors_array)
        
        final_memory = self.batch_processor.memory_monitor.get_memory_usage_mb()
        logging.info(f"Clustering completed, memory usage: {final_memory:.1f}MB")
        
        # Force garbage collection after clustering
        self.batch_processor.memory_monitor.force_garbage_collection()
        
        return {
            'labels': kmeans.labels_,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'algorithm': 'mini-batch' if use_minibatch else 'standard'
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