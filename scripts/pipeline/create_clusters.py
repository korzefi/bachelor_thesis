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
import time
import psutil
import json

import scripts.utils as utils
from scripts.utils import BatchProcessor, DataFrameChunker
from scripts.validation import validate_file_exists, validate_directory_exists


def _transform_single_file_worker(args):
    """
    Standalone worker function for multiprocessing that doesn't require pickling complex objects.

    Args:
        args: Tuple containing (period_file, periods_unique_dir, vector_temp_dir,
                               prot_vec_dict, memory_config)
    """
    period_file, periods_unique_dir, vector_temp_dir, prot_vec_dict, memory_config = args

    start_time = time.time()
    logging.info(f"Worker started: {period_file}")

    input_path = f"{periods_unique_dir}/{period_file}"
    output_path = f"{vector_temp_dir}/{period_file}"

    try:
        # Load sequences
        df = pd.read_csv(input_path)

        # Filter out invalid sequences (NaN, non-string values) and track filtered indices
        sequences = []
        filtered_indices = []
        for idx, seq in enumerate(df['sequence'].tolist()):
            if isinstance(seq, str) and seq.strip():  # Only keep non-empty strings
                sequences.append(seq)
            else:
                filtered_indices.append(idx)  # Track filtered sequence index
                if pd.isna(seq):
                    logging.warning(f"Skipping NaN sequence at index {idx} in {period_file}")
                else:
                    logging.warning(f"Skipping invalid sequence type {type(seq)} at index {idx} in {period_file}: {seq}")

        if not sequences:
            logging.warning(f"No valid sequences found in {period_file}")
            # Create empty DataFrame with proper columns
            vector_columns = [f'd{i}' for i in range(1, 101)]
            vectors_df = pd.DataFrame(columns=vector_columns)
        else:
            # Convert sequences to triplets
            triplets_list = _create_triplets_from_sequences_static(sequences)

            # Transform triplets to vectors using the dictionary
            vectors_df = _transform_triplets_to_vectors_static(triplets_list, prot_vec_dict)

        # Save vectors
        vectors_df.to_csv(output_path, index=False)

        # Save filtered indices if any sequences were filtered
        if filtered_indices:
            filtered_indices_path = f"{output_path}.filtered"
            with open(filtered_indices_path, 'w') as f:
                json.dump(filtered_indices, f)
            logging.debug(f"Saved {len(filtered_indices)} filtered indices to {filtered_indices_path}")

        elapsed_time = time.time() - start_time
        throughput = len(sequences) / elapsed_time if elapsed_time > 0 else 0

        logging.info(f"Worker completed: {period_file} - {len(sequences)} sequences → {len(vectors_df)} vectors "
                   f"in {elapsed_time:.1f}s ({throughput:.1f} seq/s)")
        return period_file, True, None

    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Worker failed: {period_file} after {elapsed_time:.1f}s - {e}")
        return period_file, False, str(e)


def _create_triplets_from_sequences_static(sequences):
    """Static function to convert sequences to 3-grams (triplets) of amino acids."""
    triplets_list = []

    for sequence in sequences:
        # Ensure sequence is a string
        if not isinstance(sequence, str):
            continue

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


def _transform_triplets_to_vectors_static(triplets_list, prot_vec_dict):
    """Static function to transform triplets to vectors using ProtVec embeddings dictionary."""
    # Create template for results (100-dimensional vectors)
    vector_columns = [f'd{i}' for i in range(1, 101)]
    result_df = pd.DataFrame(columns=vector_columns, dtype='float64')

    for i, triplets in enumerate(triplets_list):
        # Initialize sequence vector with zeros
        seq_vector = np.zeros(100, dtype='float64')

        # Sum up vectors for all triplets in the sequence
        for triplet in triplets:
            if triplet in prot_vec_dict:
                # Add the embedding vector to the sequence vector
                np.add(seq_vector, prot_vec_dict[triplet], out=seq_vector)

        # Round to 8 decimal places
        seq_vector = seq_vector.round(8)

        # Add to results
        result_df.loc[len(result_df)] = seq_vector

    return result_df


class BatchVectorProcessor(BatchProcessor):
    """Batch processor for converting sequences to vectors."""
    
    def __init__(self, config: Dict, prot_vec_df: pd.DataFrame):
        memory_config = config.get('optimization', {}).get('memory', {})
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

        # Validate required input files and directories exist
        logging.info("Validating required input files for clustering...")
        validate_file_exists(
            self.prot_vec_path,
            "ProtVec embeddings file (100d 3-grams)",
            raise_error=True
        )
        validate_directory_exists(
            self.periods_unique_dir,
            "Unique periods directory (output from prepare step)",
            raise_error=True,
            create_if_missing=False
        )
        logging.info("✓ All required input files validated")

        # Create directories
        self._create_directories()

        # Load ProtVec embeddings
        self.prot_vec_df = self._load_prot_vec_embeddings()
        
        # Initialize batch processor for vector transformation
        self.batch_processor = BatchVectorProcessor(config, self.prot_vec_df)

        # Initialize tracking variables
        self.start_time = None
        self.process = psutil.Process()

    def _log_progress(self, current: int, total: int, description: str,
                     start_time: float = None, extra_info: str = "") -> None:
        """Log progress with percentage, ETA, and optional extra info."""
        if total == 0:
            return

        percentage = (current / total) * 100

        if start_time:
            elapsed = time.time() - start_time
            if current > 0:
                eta_seconds = (elapsed / current) * (total - current)
                eta_str = f", ETA: {self._format_time(eta_seconds)}"
            else:
                eta_str = ""
            elapsed_str = f", elapsed: {self._format_time(elapsed)}"
        else:
            elapsed_str = ""
            eta_str = ""

        extra_str = f" - {extra_info}" if extra_info else ""

        logging.info(f"{description}: {current}/{total} ({percentage:.1f}%){elapsed_str}{eta_str}{extra_str}")

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _format_memory_usage(self) -> str:
        """Get formatted memory usage string."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f}MB"

    def _log_phase_summary(self, phase_name: str, stats: Dict) -> None:
        """Log a summary of completed phase with statistics."""
        logging.info(f"=== {phase_name} SUMMARY ===")
        for key, value in stats.items():
            logging.info(f"  {key}: {value}")
        logging.info(f"  Memory usage: {self._format_memory_usage()}")
        logging.info("=" * (len(phase_name) + 16))

    def _calculate_throughput(self, items_processed: int, elapsed_time: float, unit: str = "items") -> str:
        """Calculate and format throughput."""
        if elapsed_time > 0:
            throughput = items_processed / elapsed_time
            return f"{throughput:.1f} {unit}/sec"
        return "N/A"

    def _validate_and_fix_centroid_dimensions(self, centroids, expected_dims: int = 100) -> np.ndarray:
        """Validate and fix centroid dimensions to ensure consistency."""
        if centroids is None:
            return None

        # Convert to numpy array if not already
        if not isinstance(centroids, np.ndarray):
            centroids = np.array(centroids)

        # Check dimensions
        if len(centroids.shape) != 2:
            logging.error(f"Invalid centroid shape: {centroids.shape}, expected 2D array")
            return None

        n_clusters, n_dims = centroids.shape

        if n_dims == expected_dims:
            return centroids  # Already correct dimensions

        elif n_dims < expected_dims:
            # Pad with zeros
            padding = np.zeros((n_clusters, expected_dims - n_dims))
            fixed_centroids = np.hstack([centroids, padding])
            logging.warning(f"Centroid dimensions {n_dims} < {expected_dims}, padded with zeros")
            return fixed_centroids

        elif n_dims > expected_dims:
            # Truncate
            fixed_centroids = centroids[:, :expected_dims]
            logging.warning(f"Centroid dimensions {n_dims} > {expected_dims}, truncated to {expected_dims}")
            return fixed_centroids

        return centroids

    def _validate_source_period_files(self, period_files: List[str]) -> List[str]:
        """
        Validate source period files (before transformation) and return list of valid files.

        Checks that source CSV files in periods_unique_dir are:
        - Readable and accessible
        - Non-empty
        - Have correct structure (columns: isolate_name, timestamp, sequence)
        """
        valid_files = []
        invalid_files = []

        for period_file in period_files:
            source_path = f"{self.periods_unique_dir}/{period_file}"

            try:
                # Check if file exists
                if not os.path.exists(source_path):
                    invalid_files.append((period_file, "Source file does not exist"))
                    continue

                # Check file size
                file_size = os.path.getsize(source_path)
                if file_size == 0:
                    invalid_files.append((period_file, "Empty file"))
                    continue

                # Validate file structure by reading header and first row
                try:
                    sample_df = pd.read_csv(source_path, nrows=5)

                    # Check for required columns
                    required_columns = ['isolate_name', 'timestamp', 'sequence']
                    missing_columns = [col for col in required_columns if col not in sample_df.columns]

                    if missing_columns:
                        invalid_files.append((period_file, f"Missing columns: {missing_columns}"))
                        continue

                    # Check if file has any sequences
                    if len(sample_df) == 0:
                        invalid_files.append((period_file, "No sequences in file"))
                        continue

                    # File is valid
                    valid_files.append(period_file)
                    logging.debug(f"✓ Valid source file: {period_file}")

                except pd.errors.EmptyDataError:
                    invalid_files.append((period_file, "Empty or corrupted CSV"))
                except Exception as e:
                    invalid_files.append((period_file, f"Read error: {e}"))

            except Exception as e:
                invalid_files.append((period_file, f"Validation error: {e}"))

        # Log validation results
        if invalid_files:
            logging.warning(f"Found {len(invalid_files)} invalid source period files:")
            for file, reason in invalid_files[:10]:  # Show first 10
                logging.warning(f"  {file}: {reason}")
            if len(invalid_files) > 10:
                logging.warning(f"  ... and {len(invalid_files) - 10} more")

        logging.info(f"Validated {len(valid_files)} valid source files out of {len(period_files)} total")
        return valid_files

    def _validate_vector_files(self, period_files: List[str]) -> List[str]:
        """
        Validate vector files (after transformation) and return list of valid files.

        This is used as a post-transformation sanity check to ensure
        vector files were created correctly.
        """
        valid_files = []
        invalid_files = []

        for period_file in period_files:
            vector_path = f"{self.vector_temp_dir}/{period_file}"

            try:
                # Quick validation - check if file exists and has correct structure
                if not os.path.exists(vector_path):
                    invalid_files.append((period_file, "File does not exist"))
                    continue

                # Check file size
                file_size = os.path.getsize(vector_path)
                if file_size == 0:
                    invalid_files.append((period_file, "Empty file"))
                    continue

                # Read first few rows to validate structure
                try:
                    sample_df = pd.read_csv(vector_path, nrows=5, dtype='float32')

                    # Check if we have the expected number of columns (100)
                    if sample_df.shape[1] not in [99, 100, 101]:  # Allow some flexibility
                        invalid_files.append((period_file, f"Unexpected dimensions: {sample_df.shape[1]}"))
                        continue

                    # Check for NaN values
                    if sample_df.isnull().any().any():
                        logging.warning(f"Found NaN values in {period_file}")

                    valid_files.append(period_file)

                except Exception as e:
                    invalid_files.append((period_file, f"Read error: {e}"))

            except Exception as e:
                invalid_files.append((period_file, f"Validation error: {e}"))

        # Log validation results
        if invalid_files:
            logging.warning(f"Found {len(invalid_files)} invalid/missing vector files:")
            for file, reason in invalid_files[:10]:  # Show first 10
                logging.warning(f"  {file}: {reason}")
            if len(invalid_files) > 10:
                logging.warning(f"  ... and {len(invalid_files) - 10} more")

        logging.info(f"Validated {len(valid_files)} valid vector files out of {len(period_files)} total")
        return valid_files

    def _cleanup_corrupted_centroids_file(self) -> None:
        """Clean up corrupted centroids file and reinitialize."""
        if os.path.exists(self.centroids_file):
            try:
                # Try to read the file to check if it's corrupted
                test_df = pd.read_csv(self.centroids_file)
                expected_cols = 102  # period, cluster, d1...d100

                if len(test_df.columns) != expected_cols:
                    logging.warning(f"Centroids file has {len(test_df.columns)} columns, expected {expected_cols}")
                    backup_file = f"{self.centroids_file}.backup_{int(time.time())}"
                    os.rename(self.centroids_file, backup_file)
                    logging.info(f"Backed up corrupted centroids file to {backup_file}")
                    self._initialize_centroids_file()

            except Exception as e:
                logging.error(f"Centroids file appears corrupted: {e}")
                backup_file = f"{self.centroids_file}.backup_{int(time.time())}"
                try:
                    os.rename(self.centroids_file, backup_file)
                    logging.info(f"Backed up corrupted centroids file to {backup_file}")
                except:
                    logging.warning("Could not backup corrupted file, removing it")
                    os.remove(self.centroids_file)
                self._initialize_centroids_file()

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
        logging.info("=" * 60)
        logging.info("🚀 STARTING CLUSTERING PIPELINE")
        logging.info("=" * 60)

        pipeline_start_time = time.time()
        initial_memory = self._format_memory_usage()

        try:
            # Step 1: Get and validate source period files
            logging.info("📁 PHASE 1: Discovering and validating source period files...")
            logging.info(f"   Source directory: {self.periods_unique_dir}")
            phase1_start = time.time()

            period_files = self._get_period_files()
            logging.info(f"   Found {len(period_files)} period files to process")

            # Validate SOURCE files (not vector files - those don't exist yet!)
            valid_period_files = self._validate_source_period_files(period_files)

            # Clean up corrupted centroids file if needed
            self._cleanup_corrupted_centroids_file()

            phase1_elapsed = time.time() - phase1_start
            logging.info(f"✅ Phase 1 complete: {len(valid_period_files)}/{len(period_files)} valid source files in {self._format_time(phase1_elapsed)}")

            if not valid_period_files:
                logging.error("❌ No valid source period files found to process!")
                logging.error(f"   Please ensure period files exist in: {self.periods_unique_dir}")
                return

            # Use validated files for processing
            period_files = valid_period_files

            # Step 2: Transform sequences to vectors
            logging.info("\n🔄 PHASE 2: Transforming sequences to vectors...")
            logging.info(f"   Processing {len(period_files)} files")
            logging.info(f"   Output directory: {self.vector_temp_dir}")
            phase2_start = time.time()

            if self.cluster_config['use_multiprocessing']:
                self._transform_sequences_to_vectors_parallel(period_files)
            else:
                self._transform_sequences_to_vectors_sequential(period_files)

            phase2_elapsed = time.time() - phase2_start
            logging.info(f"✅ Phase 2 complete: Vector transformation finished in {self._format_time(phase2_elapsed)}")

            # Post-transformation validation (sanity check)
            logging.info("   Validating created vector files...")
            valid_vector_files = self._validate_vector_files(period_files)
            if len(valid_vector_files) < len(period_files):
                missing_count = len(period_files) - len(valid_vector_files)
                logging.warning(f"   ⚠️  {missing_count} vector files failed validation")
            else:
                logging.info(f"   ✓ All {len(valid_vector_files)} vector files validated successfully")

            # Use only successfully created vector files for clustering
            period_files = valid_vector_files

            if not period_files:
                logging.error("❌ No valid vector files after transformation!")
                logging.error("   Vector transformation may have failed. Check logs above.")
                return

            # Step 3: Create clusters for each period
            logging.info("\n🎯 PHASE 3: Clustering vectors...")
            logging.info(f"   Clustering {len(period_files)} period files")
            phase3_start = time.time()

            self._create_clusters_for_periods()

            phase3_elapsed = time.time() - phase3_start
            logging.info(f"✅ Clustering completed in {self._format_time(phase3_elapsed)}")

            # Final pipeline summary
            total_elapsed = time.time() - pipeline_start_time
            final_memory = self._format_memory_usage()

            logging.info("\n" + "=" * 60)
            logging.info("🎉 CLUSTERING PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("=" * 60)

            pipeline_stats = {
                "Files processed": len(period_files),
                "Phase 1 (Discovery)": self._format_time(phase1_elapsed),
                "Phase 2 (Vectors)": self._format_time(phase2_elapsed),
                "Phase 3 (Clustering)": self._format_time(phase3_elapsed),
                "Total pipeline time": self._format_time(total_elapsed),
                "Initial memory": initial_memory,
                "Final memory": final_memory,
                "Processing mode": "Parallel" if self.cluster_config['use_multiprocessing'] else "Sequential"
            }

            self._log_phase_summary("COMPLETE PIPELINE", pipeline_stats)

        except Exception as e:
            elapsed = time.time() - pipeline_start_time
            logging.error("💥 CLUSTERING PIPELINE FAILED!")
            logging.error(f"❌ Error after {self._format_time(elapsed)}: {e}")
            logging.error(f"🧠 Memory usage: {self._format_memory_usage()}")
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
        logging.info("=== STARTING VECTOR TRANSFORMATION (Sequential) ===")
        start_time = time.time()
        total_sequences = 0

        for i, period_file in enumerate(period_files, 1):
            file_start_time = time.time()

            # Log progress with ETA
            self._log_progress(i-1, len(period_files), "Vector transformation progress",
                             start_time, f"processing {period_file}")

            # Count sequences in file for metrics
            input_path = f"{self.periods_unique_dir}/{period_file}"
            try:
                seq_count = sum(1 for _ in open(input_path)) - 1  # Subtract header
            except:
                seq_count = 0

            self._transform_single_file(period_file)

            file_elapsed = time.time() - file_start_time
            total_sequences += seq_count

            # Log file completion with timing
            logging.info(f"Completed {period_file}: {seq_count} sequences in {self._format_time(file_elapsed)} "
                       f"({self._calculate_throughput(seq_count, file_elapsed, 'seq')})")

        # Final progress and summary
        total_elapsed = time.time() - start_time
        self._log_progress(len(period_files), len(period_files), "Vector transformation progress",
                         start_time, "COMPLETED")

        stats = {
            "Total files processed": len(period_files),
            "Total sequences": total_sequences,
            "Total time": self._format_time(total_elapsed),
            "Average throughput": self._calculate_throughput(total_sequences, total_elapsed, "seq"),
            "Average time per file": self._format_time(total_elapsed / len(period_files)) if period_files else "N/A"
        }
        self._log_phase_summary("VECTOR TRANSFORMATION", stats)
    
    def _transform_sequences_to_vectors_parallel(self, period_files: List[str]) -> None:
        """Transform sequences to vectors using multiprocessing."""
        logging.info("=== STARTING VECTOR TRANSFORMATION (Parallel) ===")
        start_time = time.time()

        try:
            # Determine number of processes
            processes = self.cluster_config.get('processes')
            if processes is None:
                processes = multiprocessing.cpu_count()

            logging.info(f"Using {processes} workers to process {len(period_files)} files")

            # Convert ProtVec DataFrame to serializable dictionary
            prep_start = time.time()
            logging.info("Preparing ProtVec data for multiprocessing...")
            prot_vec_dict = {}
            for idx, row in self.prot_vec_df.iterrows():
                vector = row.drop('words').values.astype('float64')
                prot_vec_dict[row['words']] = vector

            prep_time = time.time() - prep_start
            logging.info(f"ProtVec preparation completed in {self._format_time(prep_time)} "
                       f"({len(prot_vec_dict)} triplets)")

            # Prepare arguments for worker processes
            memory_config = self.config.get('optimization', {}).get('memory', {})
            worker_args = [
                (period_file, self.periods_unique_dir, self.vector_temp_dir,
                 prot_vec_dict, memory_config)
                for period_file in period_files
            ]

            # Create process pool and execute with progress monitoring
            logging.info(f"Starting {processes} worker processes...")
            pool_start = time.time()

            with multiprocessing.Pool(processes=processes) as pool:
                results = pool.map(_transform_single_file_worker, worker_args)

            pool_time = time.time() - pool_start

            # Process results and collect statistics
            successful_files = []
            failed_files = []
            total_sequences = 0

            for period_file, success, error in results:
                if success:
                    successful_files.append(period_file)
                    # Try to count sequences processed
                    try:
                        input_path = f"{self.periods_unique_dir}/{period_file}"
                        seq_count = sum(1 for _ in open(input_path)) - 1
                        total_sequences += seq_count
                    except:
                        pass
                else:
                    failed_files.append((period_file, error))
                    logging.error(f"Failed to process {period_file}: {error}")

            if failed_files:
                raise Exception(f"Failed to process {len(failed_files)} files in parallel")

            # Log completion statistics
            total_elapsed = time.time() - start_time

            stats = {
                "Files processed": len(successful_files),
                "Total sequences": total_sequences,
                "Workers used": processes,
                "Preparation time": self._format_time(prep_time),
                "Processing time": self._format_time(pool_time),
                "Total time": self._format_time(total_elapsed),
                "Average throughput": self._calculate_throughput(total_sequences, pool_time, "seq")
            }
            self._log_phase_summary("PARALLEL VECTOR TRANSFORMATION", stats)

        except Exception as e:
            logging.warning(f"Parallel processing failed: {e}")
            logging.info("Falling back to sequential processing...")
            self._transform_sequences_to_vectors_sequential(period_files)
    
    def _transform_single_file(self, period_file: str) -> None:
        """Transform sequences in a single file to vectors using batch processing."""
        logging.info(f"Transforming sequences for file: {period_file}")
        
        input_path = f"{self.periods_unique_dir}/{period_file}"
        output_path = f"{self.vector_temp_dir}/{period_file}"
        
        # Check if streaming is enabled
        memory_config = self.config.get('optimization', {}).get('memory', {})
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
        k_range = list(range(min_k, max_k + 1))

        logging.info(f"Finding optimal k for {n_samples} samples: testing k={min_k} to k={max_k}")
        start_time = time.time()

        vectors_array = vectors_df.to_numpy()

        for i, k in enumerate(k_range):
            k_start_time = time.time()

            # Fit K-means
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
            labels = kmeans.fit_predict(vectors_array)

            # Calculate silhouette score
            score = silhouette_score(vectors_array, labels)

            k_elapsed = time.time() - k_start_time

            # Log progress with timing
            progress_info = f"k={k} score={score:.4f} in {k_elapsed:.1f}s"
            self._log_progress(i+1, len(k_range), "K-selection progress", start_time, progress_info)

            if score > best_score:
                best_score = score
                best_k = k

        total_elapsed = time.time() - start_time
        logging.info(f"K-selection completed: optimal k={best_k} (score={best_score:.4f}) "
                   f"in {self._format_time(total_elapsed)}")
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
        logging.info("=== STARTING CLUSTERING PHASE (Sequential) ===")
        start_time = time.time()

        # Initialize centroids file
        self._initialize_centroids_file()

        # Check if auto k-selection is enabled
        auto_k_config = self.cluster_config.get('auto_k_selection', {})
        auto_k_enabled = auto_k_config.get('enabled', False)

        # Collect statistics
        total_clusters_created = 0
        total_vectors_processed = 0
        processed_periods = 0

        if auto_k_enabled:
            # Use automatic k-finding for all vector files
            vector_files = [f for f in os.listdir(self.vector_temp_dir) if f.endswith('.csv')]
            vector_files = natsorted(vector_files)

            logging.info(f"Processing {len(vector_files)} periods with automatic k-selection")

            for i, period_file in enumerate(vector_files, 1):
                period_start_time = time.time()

                # Log progress
                self._log_progress(i-1, len(vector_files), "Clustering progress",
                                 start_time, f"processing {period_file}")

                vector_file_path = f"{self.vector_temp_dir}/{period_file}"

                # Load vectors
                vectors_df = pd.read_csv(vector_file_path, dtype='float64')

                if vectors_df.empty:
                    logging.warning(f"Empty vector file: {period_file}")
                    continue

                logging.info(f"Period {period_file}: {len(vectors_df)} vectors with {vectors_df.shape[1]} dimensions")

                # Find optimal k
                n_clusters = self._find_optimal_k(vectors_df)

                # Create clusters
                cluster_results = self._create_kmeans_clusters(vectors_df, n_clusters)

                # Validate centroids before saving
                validated_centroids = self._validate_and_fix_centroid_dimensions(cluster_results['centroids'])
                if validated_centroids is not None:
                    # Save cluster labels to original period file
                    self._save_cluster_labels(period_file, cluster_results['labels'])

                    # Save centroids
                    self._save_centroids(period_file, validated_centroids)
                else:
                    logging.error(f"Failed to validate centroids for {period_file}, skipping save")
                    continue

                # Log completion statistics
                period_elapsed = time.time() - period_start_time
                total_clusters_created += n_clusters
                total_vectors_processed += len(vectors_df)
                processed_periods += 1

                logging.info(f"Completed {period_file}: {n_clusters} clusters created in {self._format_time(period_elapsed)} "
                           f"(inertia: {cluster_results.get('inertia', 'N/A'):.2f})")

        else:
            # Use manual clusters_per_period configuration (legacy mode)
            clusters_per_period = self.cluster_config.get('clusters_per_period', {})

            if not clusters_per_period:
                raise ValueError("Either enable auto_k_selection or provide clusters_per_period configuration")

            period_files = list(clusters_per_period.keys())
            logging.info(f"Processing {len(period_files)} periods with manual k configuration")

            for i, (period_file, n_clusters) in enumerate(clusters_per_period.items(), 1):
                period_start_time = time.time()

                # Log progress
                self._log_progress(i-1, len(period_files), "Clustering progress",
                                 start_time, f"processing {period_file}")

                vector_file_path = f"{self.vector_temp_dir}/{period_file}"

                if not os.path.exists(vector_file_path):
                    logging.warning(f"Vector file not found: {vector_file_path}")
                    continue

                # Load vectors
                vectors_df = pd.read_csv(vector_file_path, dtype='float64')

                if vectors_df.empty:
                    logging.warning(f"Empty vector file: {period_file}")
                    continue

                logging.info(f"Period {period_file}: {len(vectors_df)} vectors → {n_clusters} clusters")

                # Create clusters
                cluster_results = self._create_kmeans_clusters(vectors_df, n_clusters)

                # Validate centroids before saving
                validated_centroids = self._validate_and_fix_centroid_dimensions(cluster_results['centroids'])
                if validated_centroids is not None:
                    # Save cluster labels to original period file
                    self._save_cluster_labels(period_file, cluster_results['labels'])

                    # Save centroids
                    self._save_centroids(period_file, validated_centroids)
                else:
                    logging.error(f"Failed to validate centroids for {period_file}, skipping save")
                    continue

                # Log completion statistics
                period_elapsed = time.time() - period_start_time
                total_clusters_created += n_clusters
                total_vectors_processed += len(vectors_df)
                processed_periods += 1

                logging.info(f"Completed {period_file}: {n_clusters} clusters in {self._format_time(period_elapsed)} "
                           f"(inertia: {cluster_results.get('inertia', 'N/A'):.2f})")

        # Final summary
        total_elapsed = time.time() - start_time

        stats = {
            "Periods processed": processed_periods,
            "Total vectors clustered": total_vectors_processed,
            "Total clusters created": total_clusters_created,
            "Average clusters per period": f"{total_clusters_created / processed_periods:.1f}" if processed_periods > 0 else "N/A",
            "Total time": self._format_time(total_elapsed),
            "Average time per period": self._format_time(total_elapsed / processed_periods) if processed_periods > 0 else "N/A"
        }
        self._log_phase_summary("CLUSTERING PHASE", stats)

    def _create_clusters_for_periods_parallel(self) -> None:
        """Create clusters using parallel processing."""
        from joblib import Parallel, delayed

        logging.info("=== STARTING CLUSTERING PHASE (Parallel) ===")
        start_time = time.time()

        # Initialize centroids file
        self._initialize_centroids_file()

        # Get vector files
        vector_files = natsorted([f for f in os.listdir(self.vector_temp_dir) if f.endswith('.csv')])

        # Determine number of jobs
        n_jobs = min(len(vector_files), os.cpu_count())

        logging.info(f"Processing {len(vector_files)} periods using {n_jobs} parallel workers")

        # Process in parallel with timeout and memory management
        parallel_start_time = time.time()

        # Configure parallel execution with timeout and memory management
        try:
            results = Parallel(
                n_jobs=n_jobs,
                timeout=3600,  # 1 hour timeout per job
                verbose=1,     # Show progress
                backend='loky',  # Use loky backend for better timeout handling
                max_nbytes=None  # Avoid large data serialization issues
            )(
                delayed(self._process_single_period_clustering)(
                    period_file,
                    self.vector_temp_dir,
                    self.cluster_config
                ) for period_file in vector_files
            )
        except Exception as e:
            logging.error(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            logging.info("Falling back to sequential processing...")
            self._create_clusters_for_periods_sequential()
            return
        parallel_elapsed = time.time() - parallel_start_time

        # Collect and save results with statistics
        successful_periods = 0
        failed_periods = 0
        total_clusters_created = 0
        total_vectors_processed = 0

        logging.info("Collecting results from parallel workers...")

        for period_file, centroids, labels in results:
            if centroids is not None:
                # Validate centroids before saving
                validated_centroids = self._validate_and_fix_centroid_dimensions(centroids)
                if validated_centroids is not None:
                    self._save_cluster_labels(period_file, labels)
                    self._save_centroids(period_file, validated_centroids)

                    successful_periods += 1
                    total_clusters_created += len(validated_centroids)
                else:
                    failed_periods += 1
                    logging.error(f"Failed to validate centroids for {period_file}")
                    continue

                # Try to count vectors processed
                try:
                    vector_path = f"{self.vector_temp_dir}/{period_file}"
                    vectors_df = pd.read_csv(vector_path, dtype='float32')
                    total_vectors_processed += len(vectors_df)
                    logging.info(f"Saved results for {period_file}: {len(centroids)} clusters from {len(vectors_df)} vectors")
                except:
                    logging.warning(f"Could not count vectors for {period_file}")
            else:
                failed_periods += 1
                logging.error(f"Failed to process {period_file}")

        # Final summary
        total_elapsed = time.time() - start_time

        stats = {
            "Periods processed": successful_periods,
            "Failed periods": failed_periods,
            "Total vectors clustered": total_vectors_processed,
            "Total clusters created": total_clusters_created,
            "Average clusters per period": f"{total_clusters_created / successful_periods:.1f}" if successful_periods > 0 else "N/A",
            "Workers used": n_jobs,
            "Parallel processing time": self._format_time(parallel_elapsed),
            "Total time": self._format_time(total_elapsed)
        }
        self._log_phase_summary("PARALLEL CLUSTERING PHASE", stats)

    @staticmethod
    def _process_single_period_clustering(period_file: str, vector_temp_dir: str,
                                         cluster_config: dict) -> tuple:
        """Process clustering for single period with memory management."""
        import gc

        start_time = time.time()
        logging.info(f"Worker clustering started: {period_file}")

        try:
            vector_path = f"{vector_temp_dir}/{period_file}"
            vectors_df = pd.read_csv(vector_path, dtype='float32')

            if vectors_df.empty:
                logging.warning(f"Worker found empty vector file: {period_file}")
                return period_file, None, None

            logging.info(f"Worker processing {period_file}: {len(vectors_df)} vectors, {vectors_df.shape[1]} dimensions")

            # Validate vector dimensions
            expected_dims = 100
            if vectors_df.shape[1] != expected_dims:
                logging.warning(f"Worker found unexpected vector dimensions: {vectors_df.shape[1]}, expected {expected_dims}")
                if vectors_df.shape[1] < expected_dims:
                    # Pad with zeros
                    padding_cols = pd.DataFrame(0, index=vectors_df.index,
                                              columns=[f'd{i}' for i in range(vectors_df.shape[1] + 1, expected_dims + 1)])
                    vectors_df = pd.concat([vectors_df, padding_cols], axis=1)
                elif vectors_df.shape[1] > expected_dims:
                    # Truncate
                    vectors_df = vectors_df.iloc[:, :expected_dims]

            # Determine number of clusters
            if cluster_config.get('auto_k_selection', {}).get('enabled', False):
                n_clusters = ClusteringPipeline._find_optimal_k_static(vectors_df.values, cluster_config)
            else:
                n_clusters = cluster_config['clusters_per_period'].get(period_file, 5)

            # Use MiniBatchKMeans for speed
            from sklearn.cluster import MiniBatchKMeans
            # Ensure batch_size is at least 1
            batch_size = max(1, min(500, len(vectors_df) // 4))

            clustering_start = time.time()
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                n_init=3,
                max_iter=100,
                random_state=42
            )

            # Get vectors as numpy array
            vectors_array = vectors_df.values

            # Clear DataFrame to free memory
            del vectors_df
            gc.collect()

            labels = kmeans.fit_predict(vectors_array)
            clustering_time = time.time() - clustering_start

            # Validate centroid dimensions
            centroids = kmeans.cluster_centers_
            if centroids.shape[1] != expected_dims:
                logging.warning(f"Worker centroid dimension mismatch: {centroids.shape[1]}, expected {expected_dims}")

            total_elapsed = time.time() - start_time

            logging.info(f"Worker completed {period_file}: {n_clusters} clusters in {total_elapsed:.1f}s "
                       f"(clustering: {clustering_time:.1f}s, inertia: {kmeans.inertia_:.2f})")

            # Force garbage collection before returning
            gc.collect()

            return period_file, centroids, labels

        except Exception as e:
            elapsed = time.time() - start_time
            logging.error(f"Worker clustering failed for {period_file} after {elapsed:.1f}s: {e}")
            gc.collect()  # Clean up even on failure
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
        k_range = list(range(min_k, max_k + 1))

        logging.info(f"Worker k-selection: testing k={min_k} to k={max_k} on {n_samples} samples")
        start_time = time.time()

        for i, k in enumerate(k_range):
            k_start_time = time.time()

            # Fit K-means
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
            labels = kmeans.fit_predict(vectors_array)

            # Calculate silhouette score
            score = silhouette_score(vectors_array, labels)

            k_elapsed = time.time() - k_start_time
            logging.info(f"Worker k-selection: k={k} score={score:.4f} in {k_elapsed:.1f}s "
                       f"({i+1}/{len(k_range)})")

            if score > best_score:
                best_score = score
                best_k = k

        total_elapsed = time.time() - start_time
        logging.info(f"Worker k-selection completed: optimal k={best_k} (score={best_score:.4f}) "
                   f"in {total_elapsed:.1f}s")
        return best_k

    def _create_kmeans_clusters(self, vectors_df: pd.DataFrame, n_clusters: int) -> Dict:
        """Create K-means clusters from vector data with memory optimization."""
        vectors_array = vectors_df.to_numpy()
        n_samples = len(vectors_array)
        
        # Get memory optimization settings
        memory_config = self.config.get('optimization', {}).get('memory', {})
        clustering_batch_size = memory_config.get('clustering_batch_size', 500)
        
        # Choose clustering algorithm based on data size
        use_minibatch = n_samples > clustering_batch_size * 2
        
        if use_minibatch:
            logging.info(f"Using Mini-Batch K-means for {n_samples} samples (memory optimization)")
            
            # Use Mini-Batch K-means for large datasets
            # Ensure batch_size is at least 1
            batch_size = max(1, min(clustering_batch_size, n_samples // 4))
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
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
        """Save cluster labels to the original period file, handling filtered sequences."""
        original_file_path = f"{self.periods_unique_dir}/{period_file}"
        vector_path = f"{self.vector_temp_dir}/{period_file}"
        filtered_indices_path = f"{vector_path}.filtered"

        # Load original data
        df = pd.read_csv(original_file_path)

        # Load filtered indices if they exist
        filtered_indices = []
        if os.path.exists(filtered_indices_path):
            with open(filtered_indices_path, 'r') as f:
                filtered_indices = json.load(f)

        # Create full labels array with -1 for filtered sequences
        full_labels = np.full(len(df), -1, dtype=int)

        # Assign cluster labels to valid sequences
        valid_idx = 0
        for i in range(len(df)):
            if i not in filtered_indices:
                if valid_idx < len(labels):
                    full_labels[i] = labels[valid_idx]
                    valid_idx += 1

        # Remove existing cluster column if it exists
        if 'cluster' in df.columns:
            df.drop('cluster', axis=1, inplace=True)

        # Add cluster labels
        df.insert(0, 'cluster', full_labels)

        # Save back to file
        df.to_csv(original_file_path, index=False)

        logging.debug(f"Added cluster labels to {period_file} (assigned {valid_idx} clusters, {len(filtered_indices)} filtered)")
    
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
        """Save cluster centroids to the centroids file with robust validation using append mode."""
        period_name = period_file[:-4]  # Remove .csv extension

        try:
            # Validate and fix centroid dimensions
            validated_centroids = self._validate_and_fix_centroid_dimensions(centroids)
            if validated_centroids is None:
                logging.error(f"Failed to validate centroids for {period_name}")
                return

            # Prepare rows for writing
            new_rows = []
            for cluster_num, centroid in enumerate(validated_centroids):
                # Ensure centroid is a 1D array and convert to list
                if len(centroid.shape) > 1:
                    centroid = centroid.flatten()

                centroid_list = centroid.tolist()

                # Ensure exactly 100 dimensions
                if len(centroid_list) < 100:
                    centroid_list.extend([0.0] * (100 - len(centroid_list)))
                elif len(centroid_list) > 100:
                    centroid_list = centroid_list[:100]

                # Create row: period, cluster, d1, d2, ..., d100
                row_data = [period_name, cluster_num] + centroid_list

                # Validate row has exactly 102 elements
                if len(row_data) != 102:
                    logging.error(f"Row data length {len(row_data)} != 102 for {period_name}")
                    continue

                new_rows.append(row_data)

            if not new_rows:
                logging.warning(f"No valid centroids to save for {period_name}")
                return

            # Use a simpler append-based approach to avoid DataFrame concatenation issues
            file_exists = os.path.exists(self.centroids_file)

            if not file_exists:
                # File doesn't exist, create it with header
                self._initialize_centroids_file()

            # Append new rows directly to the CSV file
            with open(self.centroids_file, 'a', newline='') as f:
                import csv
                writer = csv.writer(f)
                for row in new_rows:
                    writer.writerow(row)

            logging.info(f"Saved {len(new_rows)} centroids for {period_name}")

        except Exception as e:
            logging.error(f"Failed to save centroids for {period_name}: {e}")
            logging.error(f"Centroids shape: {centroids.shape if hasattr(centroids, 'shape') else 'unknown'}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise - continue with other files


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