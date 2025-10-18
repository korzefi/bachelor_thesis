#!/usr/bin/env python3
"""
Dataset Creation Pipeline Module

This module handles the creation of the final training dataset from linked clusters by:
1. Creating sliding windows from consecutive time periods
2. Linking clusters across periods based on centroids
3. Extracting epitope regions with context
4. Converting sequences to ProtVec indices
5. Creating binary classification labels (mutated/not mutated)
6. Refilling dataset to achieve target mutation ratio

Refactored from: scripts/preprocessing/EpitopeDataCreator.py
"""

import logging
import os
import random
import pandas as pd
from pathlib import Path
from natsort import natsorted
from sklearn.utils import shuffle
from typing import Dict, List, Tuple, Any
import multiprocessing
import pickle
import itertools
import time
from joblib import Parallel, delayed

import scripts.utils as utils
from scripts.utils import BatchProcessor, DataFrameChunker
from scripts.validation import validate_file_exists, validate_directory_exists


class BatchDatasetProcessor(BatchProcessor):
    """Batch processor for streaming dataset creation."""

    def __init__(self, config: Dict, epitopes_positions: List[int], window_size: int):
        memory_config = config.get('optimization', {}).get('memory', {})
        batch_size = memory_config.get('dataset_creation_batch_size', 100)
        super().__init__(config, batch_size, "BatchDatasetProcessor")

        self.epitopes_positions = epitopes_positions
        self.window_size = window_size
        self.output_file = None
        self.window_metadata = None  # Store metadata for period tracking

        # ProtVec batch size for transformation
        self.protvec_batch_size = memory_config.get('protvec_batch_size', 50)
    
    def setup_output_file(self, output_path: str):
        """Setup output file for streaming results."""
        self.output_file = output_path

        # Remove existing output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

    def set_metadata(self, window_metadata: List[Dict]):
        """Set the metadata for period tracking."""
        self.window_metadata = window_metadata

    def process_in_batches(self, data_source: Any, description: str = "Processing") -> list:
        """Override to handle metadata alignment with batches."""
        logging.info(f"{self.component_name}: Starting {description} with batch size {self.batch_size}")

        batch_count = 0

        try:
            for batch_index, batch_data in enumerate(self.create_data_iterator(data_source)):
                # Log progress
                if batch_index % 10 == 0:
                    memory_mb = self.memory_monitor.get_memory_usage_mb()
                    logging.info(f"{self.component_name}: Processing batch {batch_index + 1}, Memory: {memory_mb:.1f}MB")

                # Extract metadata for this batch (if available)
                batch_start_idx = batch_index * self.batch_size
                batch_end_idx = batch_start_idx + len(batch_data)
                if self.window_metadata and batch_start_idx < len(self.window_metadata):
                    batch_metadata = self.window_metadata[batch_start_idx:batch_end_idx]
                else:
                    batch_metadata = None

                # Process batch with metadata
                batch_result = self.process_batch_with_metadata(batch_data, batch_metadata)

                # Save result
                if self.intermediate_saves:
                    self.save_batch_result(batch_result, batch_index)

                # Memory monitoring
                self.memory_monitor.batch_completed()
                batch_count += 1

        except Exception as e:
            logging.error(f"{self.component_name}: Batch processing failed at batch {batch_count}: {e}")
            raise

        logging.info(f"{self.component_name}: Completed {description}, processed {batch_count} batches")
        return []

    def process_batch_with_metadata(self, batch_samples: List[List[str]],
                                   batch_metadata: List[Dict] = None) -> pd.DataFrame:
        """Process a batch of sequence samples with metadata into dataset format."""
        # Extract epitopes with context from batch
        epitope_samples = self._extract_epitopes_with_context_batch(batch_samples)

        # Transform to ProtVec indices in sub-batches
        protvec_samples = self._transform_to_protvec_indices_batch(epitope_samples)

        # Create final dataframe from batch with metadata
        dataset_df = self._create_dataframe_from_batch(protvec_samples, batch_metadata)

        return dataset_df

    def process_batch(self, batch_samples: List[List[str]]) -> pd.DataFrame:
        """Process a batch of sequence samples into dataset format (backward compatibility)."""
        return self.process_batch_with_metadata(batch_samples, None)
    
    def save_batch_result(self, dataset_df: pd.DataFrame, batch_index: int) -> None:
        """Save batch result incrementally to final output file."""
        if self.output_file and not dataset_df.empty:
            DataFrameChunker.write_csv_incrementally(
                dataset_df, 
                self.output_file, 
                mode='a'
            )
    
    def _extract_epitopes_with_context_batch(self, batch_samples: List[List[str]]) -> List[List[List[str]]]:
        """Extract epitope regions with context from batch of samples."""
        epitope_samples = []
        
        for sample in batch_samples:
            sample_epitopes = []
            for sequence in sample:
                # Remove asterisk if present
                if sequence.endswith('*'):
                    sequence = sequence[:-1]
                
                sequence_epitopes = []
                for position in self.epitopes_positions:
                    context_size = 2  # From config
                    start_pos = max(0, position - context_size)
                    end_pos = min(len(sequence), position + context_size + 1)
                    epitope_context = sequence[start_pos:end_pos]
                    sequence_epitopes.append(epitope_context)
                
                sample_epitopes.append(sequence_epitopes)
            epitope_samples.append(sample_epitopes)
        
        return epitope_samples
    
    def _transform_to_protvec_indices_batch(self, epitope_samples: List[List[List[str]]]) -> List[List[List[List[int]]]]:
        """Transform epitope contexts to ProtVec indices in small sub-batches."""
        protvec_samples = []
        
        # Process in smaller sub-batches to control memory
        for i in range(0, len(epitope_samples), self.protvec_batch_size):
            batch_end = min(i + self.protvec_batch_size, len(epitope_samples))
            sub_batch = epitope_samples[i:batch_end]
            
            for sample_idx, sample in enumerate(sub_batch):
                sample_protvec = []
                for sequence_epitopes in sample:
                    sequence_protvec = []
                    for epitope_context in sequence_epitopes:
                        # Create triplets from epitope context
                        triplets = self._create_triplets_from_epitope(epitope_context)
                        # Convert triplets to ProtVec indices
                        triplet_indices = self._triplets_to_indices(triplets)
                        sequence_protvec.append(triplet_indices)
                    sample_protvec.append(sequence_protvec)
                protvec_samples.append(sample_protvec)
        
        return protvec_samples
    
    def _create_dataframe_from_batch(self, protvec_samples: List[List[List[List[int]]]],
                                    batch_metadata: List[Dict] = None) -> pd.DataFrame:
        """Create DataFrame from batch of ProtVec samples with metadata."""
        if not protvec_samples:
            columns = ['y'] + [str(i) for i in range(self.window_size)] + ['period_start', 'period_end']
            return pd.DataFrame(columns=columns)

        # Pre-allocate list for rows
        all_rows = []

        for sample_idx, sample in enumerate(protvec_samples):
            epitopes_count = len(sample[0])  # Number of epitope positions

            # Get metadata for this sample if available
            if batch_metadata and sample_idx < len(batch_metadata):
                period_info = batch_metadata[sample_idx]
                period_start = period_info.get('period_start', '')
                period_end = period_info.get('period_end', '')
            else:
                period_start = ''
                period_end = ''

            for epitope_idx in range(epitopes_count):
                row_data = []

                # Collect data for this epitope position across all sequences in the sample
                for sequence_idx in range(len(sample)):
                    if epitope_idx < len(sample[sequence_idx]):
                        row_data.append(sample[sequence_idx][epitope_idx])
                    else:
                        row_data.append([])  # Empty if epitope not available

                # Create row for dataset with metadata
                dataset_row = self._create_dataset_row(row_data)
                dataset_row.extend([period_start, period_end])
                all_rows.append(dataset_row)

        # Create DataFrame from all rows with period columns
        columns = ['y'] + [str(i) for i in range(self.window_size)] + ['period_start', 'period_end']
        return pd.DataFrame(all_rows, columns=columns)
    
    def _create_triplets_from_epitope(self, epitope_context: str) -> List[str]:
        """Create 3-grams from epitope context."""
        context_size = 2  # From config
        sites_per_position = 1 + (2 * context_size)
        triplets_num = sites_per_position - 2
        
        if len(epitope_context) < 3:
            return []
        
        triplets = [epitope_context[i:i+3] for i in range(min(triplets_num, len(epitope_context) - 2))]
        return triplets
    
    def _triplets_to_indices(self, triplets: List[str]) -> List[int]:
        """Convert triplets to ProtVec indices using lookup (placeholder)."""
        # This will be initialized with triplet_to_index from parent class
        indices = []
        for triplet in triplets:
            if hasattr(self, 'triplet_to_index') and triplet in self.triplet_to_index:
                indices.append(self.triplet_to_index[triplet])
            else:
                indices.append(0)  # Default for unknown triplets
        return indices
    
    def _create_dataset_row(self, row_data: List[List[int]]) -> List:
        """Create a single dataset row with mutation label."""
        # Separate target (y) from input sequences (x)
        input_sequences = row_data[:-1]
        target_sequence = row_data[-1]
        
        # Create mutation label
        if len(input_sequences) > 0:
            last_input_sequence = input_sequences[-1]
            # Check if target is similar to last input (mutation detection)
            mutation_label = 0 if self._sequences_similar(last_input_sequence, target_sequence) else 1
        else:
            mutation_label = 0
        
        # Convert lists to strings for DataFrame compatibility
        input_sequences_str = [str(seq) for seq in input_sequences]
        
        # Create final row
        result_row = [mutation_label] + input_sequences_str
        return result_row
    
    def _sequences_similar(self, seq1: List[int], seq2: List[int]) -> bool:
        """Check if two sequences are similar (have 3 or more matching elements)."""
        if not seq1 or not seq2:
            return True
        
        matching_elements = len(set(seq1) & set(seq2))
        return matching_elements >= 3


class DatasetCreationError(Exception):
    """Custom exception for dataset creation errors."""
    pass


class SequencesMutatedTooMuch(DatasetCreationError):
    """Exception raised when sequences are mutated beyond threshold."""
    def __init__(self, message="Could not find sequences fulfilling threshold criterion"):
        self.message = message
        super().__init__(self.message)


class CheckpointManager:
    """Manages checkpointing for resumable dataset creation."""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.data = self._load_checkpoint()

        # Ensure output directory exists
        Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self) -> Dict:
        """Load existing checkpoint data or create new one."""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                logging.info(f"Loaded checkpoint from {self.checkpoint_file}")
                return data
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

        return {
            'completed_windows': {},
            'last_completed_window': -1,
            'total_samples_created': 0,
            'start_time': time.time()
        }

    def save_window_results(self, window_idx: int, samples: List) -> None:
        """Save completed window results."""
        self.data['completed_windows'][window_idx] = {
            'samples': samples,
            'timestamp': time.time(),
            'sample_count': len(samples)
        }
        self.data['last_completed_window'] = max(
            self.data['last_completed_window'],
            window_idx
        )
        self.data['total_samples_created'] += len(samples)

        # Save to disk
        self._save_checkpoint()

        logging.info(f"Checkpoint saved: Window {window_idx} completed with {len(samples)} samples")

    def save_batch_results(self, window_results: List[Tuple[int, List]]) -> None:
        """Save a batch of window results."""
        for window_idx, samples in window_results:
            self.save_window_results(window_idx, samples)

    def get_resume_point(self) -> int:
        """Get the window index to resume from."""
        return self.data['last_completed_window'] + 1

    def get_completed_samples(self) -> List:
        """Get all completed samples for final processing."""
        all_samples = []
        for window_idx in sorted(self.data['completed_windows'].keys()):
            window_data = self.data['completed_windows'][window_idx]
            all_samples.extend(window_data['samples'])
        return all_samples

    def _save_checkpoint(self) -> None:
        """Save checkpoint data to disk."""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.data, f)
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    def clear_checkpoint(self) -> None:
        """Clear checkpoint file after successful completion."""
        try:
            if Path(self.checkpoint_file).exists():
                os.remove(self.checkpoint_file)
                logging.info("Checkpoint file cleared after successful completion")
        except Exception as e:
            logging.warning(f"Failed to clear checkpoint file: {e}")

    def get_progress_info(self) -> Dict:
        """Get progress information for logging."""
        elapsed_time = time.time() - self.data['start_time']
        return {
            'completed_windows': len(self.data['completed_windows']),
            'total_samples': self.data['total_samples_created'],
            'elapsed_time': elapsed_time,
            'last_window': self.data['last_completed_window']
        }


class SequenceCache:
    """Pre-loads and caches all sequences for fast retrieval."""

    def __init__(self, periods_dir: str):
        self.periods_dir = periods_dir
        self.cache = {}  # (period, cluster) -> [sequences]
        self.period_files = {}  # period -> DataFrame
        self._preload_all_sequences()

    def _preload_all_sequences(self) -> None:
        """Pre-load all period files and build cluster→sequences cache."""
        start_time = time.time()

        period_files = [f for f in os.listdir(self.periods_dir) if f.endswith('.csv')]
        logging.info(f"Pre-loading sequences from {len(period_files)} period files...")

        total_sequences = 0
        total_clusters = 0

        for period_file in period_files:
            period_name = period_file.replace('.csv', '')
            file_path = os.path.join(self.periods_dir, period_file)

            try:
                df = pd.read_csv(file_path)

                # Store the DataFrame for quick access
                self.period_files[period_name] = df

                # Group sequences by cluster
                for cluster_id in df['cluster'].unique():
                    # Skip invalid clusters (e.g., -1 for filtered sequences)
                    if cluster_id < 0:
                        continue

                    cluster_sequences = df[df['cluster'] == cluster_id]['sequence'].tolist()
                    # Filter out NaN sequences
                    cluster_sequences = [seq for seq in cluster_sequences if pd.notna(seq)]

                    if cluster_sequences:
                        self.cache[(period_name, cluster_id)] = cluster_sequences
                        total_sequences += len(cluster_sequences)
                        total_clusters += 1

            except Exception as e:
                logging.warning(f"Failed to load period file {period_file}: {e}")

        elapsed_time = time.time() - start_time
        logging.info(f"Sequence cache loaded in {elapsed_time:.1f}s: "
                    f"{total_sequences:,} sequences across {total_clusters:,} clusters")

    def get_sequences_for_cluster(self, period: str, cluster: int) -> List[str]:
        """Get cached sequences for a cluster."""
        return self.cache.get((period, cluster), [])

    def get_random_sequence_from_cluster(self, period: str, cluster: int) -> str:
        """Get a random sequence from a specific cluster."""
        sequences = self.get_sequences_for_cluster(period, cluster)

        if not sequences:
            raise DatasetCreationError(f"No sequences found for cluster {cluster} in period {period}")

        return random.choice(sequences)

    def get_cluster_count(self, period: str) -> int:
        """Get the number of clusters in a period."""
        if period in self.period_files:
            return len(self.period_files[period]['cluster'].unique())
        return 0

    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring."""
        total_sequences = sum(len(seqs) for seqs in self.cache.values())
        total_clusters = len(self.cache)
        periods_count = len(set(period for period, _ in self.cache.keys()))

        return {
            'total_sequences': total_sequences,
            'total_clusters': total_clusters,
            'periods_count': periods_count,
            'avg_sequences_per_cluster': total_sequences / total_clusters if total_clusters > 0 else 0
        }


def _process_window_worker(args):
    """Worker function for parallel window processing."""
    (window_idx, window, samples_per_window, centroids_df_dict,
     sequence_cache_data, pipeline_config) = args

    # Reconstruct objects from serializable data
    centroids_df = pd.DataFrame(centroids_df_dict)

    # Worker-specific logging
    worker_start = time.time()
    logging.info(f"Worker starting window {window_idx}: {window}")

    # Create samples in parallel within this window using joblib
    sample_workers = pipeline_config.get('sample_workers', 2)
    try:
        samples = Parallel(n_jobs=sample_workers, backend='threading')(
            delayed(_create_single_sample_cached)(
                centroids_df, window, sequence_cache_data, pipeline_config
            ) for _ in range(samples_per_window)
        )

        # Filter out failed samples (None values)
        successful_samples = [s for s in samples if s is not None]

        worker_time = time.time() - worker_start
        logging.info(f"Worker completed window {window_idx}: {len(successful_samples)}/{samples_per_window} samples in {worker_time:.1f}s")

        return window_idx, successful_samples

    except Exception as e:
        logging.error(f"Worker failed for window {window_idx}: {e}")
        return window_idx, []


def _create_single_sample_cached(centroids_df, window, sequence_cache_data, pipeline_config):
    """Create a single sequence sample using cached sequences."""
    max_retries = 10

    for retry in range(max_retries):
        try:
            # Choose first sequence randomly
            first_sequence_info = _choose_first_sequence_cached(
                centroids_df, window, sequence_cache_data
            )

            # Link remaining sequences
            sample_sequences = _link_remaining_sequences_cached(
                centroids_df, first_sequence_info, window,
                sequence_cache_data, pipeline_config
            )

            return sample_sequences

        except SequencesMutatedTooMuch:
            if retry == max_retries - 1:
                logging.warning(f"Sample creation failed after {max_retries} retries (mutation threshold)")
                return None
            continue
        except Exception as e:
            logging.warning(f"Sample creation failed (retry {retry + 1}): {e}")
            if retry == max_retries - 1:
                return None
            continue

    return None


def _choose_first_sequence_cached(centroids_df, window, sequence_cache_data):
    """Choose the first sequence using cached data."""
    first_period = window['x'][0]
    first_period_rows = centroids_df[centroids_df['period'] == first_period]

    if first_period_rows.empty:
        raise DatasetCreationError(f"No clusters found for period: {first_period}")

    # Sample random cluster
    chosen_row = first_period_rows.sample(n=1).iloc[0]
    current_cluster = chosen_row['cluster']
    next_clusters_str = chosen_row['next_cluster']

    # Parse next clusters
    if pd.isna(next_clusters_str) or next_clusters_str == '':
        next_clusters = []
    else:
        next_clusters = [int(x) for x in str(next_clusters_str).split('-') if x.strip()]

    # Get sequence from cache
    cache_key = (first_period, current_cluster)
    if cache_key in sequence_cache_data:
        sequences = sequence_cache_data[cache_key]
        if sequences:
            sequence = random.choice(sequences)
        else:
            raise DatasetCreationError(f"No sequences in cache for cluster {current_cluster} in period {first_period}")
    else:
        raise DatasetCreationError(f"Cache key {cache_key} not found")

    return {
        'sequence': sequence,
        'next_clusters': next_clusters
    }


def _link_remaining_sequences_cached(centroids_df, first_sequence_info, window,
                                   sequence_cache_data, pipeline_config):
    """Link remaining sequences using cached data."""
    sequences = [first_sequence_info['sequence']]
    current_next_clusters = first_sequence_info['next_clusters']

    # Get pipeline configuration
    epitopes_positions = pipeline_config.get('epitopes_positions', [])
    epitopes_similarity_threshold = pipeline_config.get('epitopes_similarity_threshold', 0.5)
    max_similar_epitopes = int(len(epitopes_positions) * epitopes_similarity_threshold)

    # Process x periods (input sequence)
    for i in range(1, len(window['x'])):
        current_period = window['x'][i]

        if not current_next_clusters:
            raise DatasetCreationError(f"No next clusters available for period: {current_period}")

        # Choose random cluster from available next clusters
        current_cluster = random.choice(current_next_clusters)

        # Get cluster info for current period
        cluster_row = centroids_df[
            (centroids_df['period'] == current_period) &
            (centroids_df['cluster'] == current_cluster)
        ]

        if cluster_row.empty:
            raise DatasetCreationError(f"Cluster {current_cluster} not found in period {current_period}")

        # Get sequence from cache and check mutation threshold
        cache_key = (current_period, current_cluster)
        if cache_key not in sequence_cache_data:
            raise DatasetCreationError(f"Cache key {cache_key} not found")

        cached_sequences = sequence_cache_data[cache_key]
        if not cached_sequences:
            raise DatasetCreationError(f"No sequences in cache for cluster {current_cluster} in period {current_period}")

        previous_sequence = sequences[-1]

        # Retry if mutated too much
        max_mutation_retries = 10
        retry_count = 0
        sequence = None

        while retry_count < max_mutation_retries:
            sequence = random.choice(cached_sequences)
            if not _is_mutated_too_much_cached(previous_sequence, sequence, epitopes_positions, max_similar_epitopes):
                break
            retry_count += 1

        if retry_count >= max_mutation_retries:
            raise SequencesMutatedTooMuch()

        sequences.append(sequence)

        # Update next clusters for next iteration
        next_clusters_str = cluster_row.iloc[0]['next_cluster']
        if pd.isna(next_clusters_str) or next_clusters_str == '':
            current_next_clusters = []
        else:
            current_next_clusters = [int(x) for x in str(next_clusters_str).split('-') if x.strip()]

    # Add y period sequence (target)
    y_period = window['y']
    if current_next_clusters:
        target_cluster = random.choice(current_next_clusters)
        cache_key = (y_period, target_cluster)
        if cache_key in sequence_cache_data and sequence_cache_data[cache_key]:
            target_sequence = random.choice(sequence_cache_data[cache_key])
            sequences.append(target_sequence)
        else:
            raise DatasetCreationError(f"No sequences in cache for cluster {target_cluster} in period {y_period}")
    else:
        raise DatasetCreationError(f"No clusters available for target period: {y_period}")

    return sequences


def _is_mutated_too_much_cached(prev_sequence, current_sequence, epitopes_positions, max_similar_epitopes):
    """Check if sequences are mutated beyond threshold in epitope regions."""
    mutated_epitopes = 0

    for pos in epitopes_positions:
        if pos < len(prev_sequence) and pos < len(current_sequence):
            if prev_sequence[pos] != current_sequence[pos]:
                mutated_epitopes += 1

    return mutated_epitopes > max_similar_epitopes


def _transform_protvec_chunk_worker(args):
    """Worker function for parallel ProtVec transformation."""
    chunk_samples, triplet_to_index, epitopes_positions, context_size, window_size = args

    result_samples = []
    for sample in chunk_samples:
        # Process each sample
        sample_result = _transform_single_sample_protvec(
            sample, triplet_to_index, epitopes_positions, context_size, window_size
        )
        result_samples.append(sample_result)

    return result_samples


def _transform_single_sample_protvec(sample, triplet_to_index, epitopes_positions, context_size, window_size):
    """Transform a single sample to ProtVec indices."""
    sample_protvec = []

    for sequence in sample:
        # Remove asterisk if present
        if sequence.endswith('*'):
            sequence = sequence[:-1]

        sequence_protvec = []
        for position in epitopes_positions:
            start_pos = max(0, position - context_size)
            end_pos = min(len(sequence), position + context_size + 1)
            epitope_context = sequence[start_pos:end_pos]

            # Create triplets from epitope context
            sites_per_position = 1 + (2 * context_size)
            triplets_num = sites_per_position - 2

            if len(epitope_context) < 3:
                triplet_indices = []
            else:
                triplets = [epitope_context[i:i+3] for i in range(min(triplets_num, len(epitope_context) - 2))]
                # Convert triplets to ProtVec indices
                triplet_indices = []
                for triplet in triplets:
                    if triplet in triplet_to_index:
                        triplet_indices.append(triplet_to_index[triplet])
                    else:
                        triplet_indices.append(0)  # Default for unknown triplets

            sequence_protvec.append(triplet_indices)
        sample_protvec.append(sequence_protvec)

    return sample_protvec


class DatasetCreationPipeline:
    """Main class that orchestrates the dataset creation process."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration dictionary."""
        self.config = config
        self.data_config = config['data']
        self.dataset_config = config['create_dataset']
        
        # Core parameters
        self.window_size = self.dataset_config['window_size']
        self.context_size = self.dataset_config['context_size']
        self.epitopes_similarity_threshold = self.dataset_config['epitopes_similarity_threshold']
        self.dataset_size = self.dataset_config['dataset_size']

        # Duplication factor to compensate for duplicate removal
        self.duplication_factor = self.dataset_config.get('duplication_factor', 20)

        # Parse epitope positions
        self.epitopes_positions = self._parse_epitope_positions(self.dataset_config['epitopes'])
        self.max_similar_epitopes = int(len(self.epitopes_positions) * self.epitopes_similarity_threshold)

        # Validate dataset_size for small values
        min_dataset_size = len(self.epitopes_positions)
        if self.dataset_size < min_dataset_size:
            logging.warning(f"dataset_size ({self.dataset_size}) is smaller than epitope positions ({min_dataset_size}). "
                          f"This targets {self.dataset_size} rows in the FINAL dataset after deduplication.")
            logging.warning(f"Consider setting dataset_size to at least {min_dataset_size} for a minimal final dataset, "
                          f"or {min_dataset_size * 10} for a small but functional dataset.")
            logging.info(f"Note: dataset_size controls the FINAL size after duplicate removal, not the initial generation.")
        
        # File paths
        self.linked_centroids_file = self.data_config['linked_centroids_file']
        self.periods_unique_dir = self.data_config['periods_unique_dir']
        self.prot_vec_file = self.data_config['prot_vec_file']
        self.final_dataset_file = self.data_config['final_dataset_file']

        # Validate required input files and directories exist
        logging.info("Validating required input files for dataset creation...")
        validate_file_exists(
            self.prot_vec_file,
            "ProtVec embeddings file (100d 3-grams)",
            raise_error=True
        )
        validate_file_exists(
            self.linked_centroids_file,
            "Linked cluster centroids file (output from cluster step)",
            raise_error=True
        )
        validate_directory_exists(
            self.periods_unique_dir,
            "Unique periods directory (output from prepare step)",
            raise_error=True,
            create_if_missing=False
        )
        logging.info("✓ All required input files validated")

        # Parallelization configuration
        self.parallel_config = self.dataset_config.get('parallel', {})
        self.parallel_windows = self.parallel_config.get('enabled', True)
        self.window_workers = self.parallel_config.get('window_workers', min(8, multiprocessing.cpu_count() - 1))
        self.sample_workers = self.parallel_config.get('sample_workers', 2)

        # Checkpointing configuration
        checkpoint_config = self.dataset_config.get('checkpoint', {})
        self.checkpoint_enabled = checkpoint_config.get('enabled', True)
        checkpoint_file = checkpoint_config.get('file', 'data/processed/dataset_checkpoint.pkl')
        self.checkpoint_interval = checkpoint_config.get('interval', 10)

        # Initialize checkpoint manager
        if self.checkpoint_enabled:
            self.checkpoint_manager = CheckpointManager(checkpoint_file)
        else:
            self.checkpoint_manager = None

        # Load ProtVec embeddings
        self.prot_vec_df = self._load_prot_vec_embeddings()

        # Initialize sequence cache (if enabled)
        cache_config = self.dataset_config.get('cache', {})
        self.cache_enabled = cache_config.get('enabled', True)
        if self.cache_enabled:
            logging.info("Initializing sequence cache...")
            self.sequence_cache = SequenceCache(self.periods_unique_dir)
            cache_stats = self.sequence_cache.get_cache_stats()
            logging.info(f"Sequence cache initialized: {cache_stats}")
        else:
            self.sequence_cache = None

        # Initialize batch processor for streaming dataset creation
        self.batch_dataset_processor = BatchDatasetProcessor(
            config, self.epitopes_positions, self.window_size
        )

        # Share ProtVec lookup with batch processor
        self.batch_dataset_processor.triplet_to_index = self.triplet_to_index

        # Create output directory
        self._create_output_directory()
    
    def _parse_epitope_positions(self, epitope_ranges: List[List[int]]) -> List[int]:
        """Parse epitope ranges into flat list of positions (0-based indexing)."""
        epitopes = []
        for start, end in epitope_ranges:
            epitopes.extend(list(range(start, end + 1)))
        
        # Convert to 0-based indexing
        epitopes = [pos - 1 for pos in epitopes]
        return epitopes
    
    def _load_prot_vec_embeddings(self) -> pd.DataFrame:
        """Load ProtVec embeddings."""
        logging.info(f"Loading ProtVec embeddings from: {self.prot_vec_file}")
        df = pd.read_csv(self.prot_vec_file, sep='\t')
        
        # Create lookup dictionary for faster triplet-to-index mapping
        self.triplet_to_index = {}
        for idx, row in df.iterrows():
            self.triplet_to_index[row['words']] = idx
        
        logging.info(f"Created lookup dictionary with {len(self.triplet_to_index)} triplets")
        return df
    
    def _create_output_directory(self) -> None:
        """Create output directory for the dataset."""
        output_dir = Path(self.final_dataset_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> None:
        """Execute the complete dataset creation pipeline."""
        logging.info("Starting dataset creation pipeline...")
        
        try:
            # Check if refilling is enabled
            refiller_config = self.dataset_config.get('refiller', {})
            if refiller_config.get('enabled', False):
                self._run_with_refilling()
            else:
                self._run_single_pass()
            
            logging.info("Dataset creation pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Dataset creation pipeline failed: {e}")
            raise
    
    def _run_single_pass(self) -> None:
        """Run dataset creation without refilling."""
        # Get expected duplicate rate for better estimation
        expected_dup_rate = self.dataset_config.get('expected_duplicate_rate', 0.8)

        # Log the target
        logging.info(f"=== SINGLE PASS MODE ===")
        logging.info(f"Target final dataset size (after dedup): {self.dataset_size} rows")
        logging.info(f"Expected duplicate rate: {expected_dup_rate:.1%}")

        dataset_df = self._create_dataset()
        dataset_df, duplicate_rate = self._remove_duplicates(dataset_df)

        # Check if we reached target
        total_samples = len(dataset_df)
        target_reached_percent = (total_samples / self.dataset_size * 100) if self.dataset_size > 0 else 0

        if total_samples < self.dataset_size * 0.95:
            logging.warning(f"Dataset size ({total_samples}) is below 95% of target ({self.dataset_size})")
            logging.warning(f"Only achieved {target_reached_percent:.1f}% of target size")
            logging.warning(f"Actual duplicate rate ({duplicate_rate:.1%}) vs expected ({expected_dup_rate:.1%})")
            logging.warning(f"Consider enabling refiller or adjusting duplication_factor/expected_duplicate_rate")
        else:
            logging.info(f"Successfully reached {target_reached_percent:.1f}% of target size")

        self._save_dataset(dataset_df)

        # Log final statistics
        mutated_samples = len(dataset_df[dataset_df['y'] == 1])
        mutation_ratio = mutated_samples / total_samples if total_samples > 0 else 0

        logging.info(f"=== FINAL DATASET STATISTICS ===")
        logging.info(f"Target size: {self.dataset_size} rows")
        logging.info(f"Actual size: {total_samples} rows ({target_reached_percent:.1f}% of target)")
        logging.info(f"Mutation ratio: {mutation_ratio:.3f} ({mutated_samples}/{total_samples})")
        logging.info(f"Duplicate rate: {duplicate_rate:.1%}")

        # Clear checkpoint after successful completion
        if self.checkpoint_manager:
            self.checkpoint_manager.clear_checkpoint()
    
    def _run_with_refilling(self) -> None:
        """Run dataset creation with iterative refilling and early stopping."""
        refiller_config = self.dataset_config['refiller']
        max_iterations = refiller_config.get('max_iterations', 20)
        target_ratio = refiller_config.get('target_mutated_ratio', 0.2)
        early_stopping_threshold = refiller_config.get('early_stopping_threshold', 0.95)
        expected_dup_rate = self.dataset_config.get('expected_duplicate_rate', 0.8)

        logging.info(f"=== REFILLING MODE ===")
        logging.info(f"Target FINAL dataset size (after dedup): {self.dataset_size} rows")
        logging.info(f"Target mutation ratio: {target_ratio:.1%}")
        logging.info(f"Early stopping threshold: {early_stopping_threshold:.1%} of target")
        logging.info(f"Expected duplicate rate: {expected_dup_rate:.1%}")

        dataset_df = self._create_dataset()
        dataset_df, initial_duplicate_rate = self._remove_duplicates(dataset_df)

        duplicate_rates = [initial_duplicate_rate]
        no_improvement_count = 0
        best_size = len(dataset_df)

        for iteration in range(1, max_iterations + 1):
            total_samples = len(dataset_df)
            mutated_samples = len(dataset_df[dataset_df['y'] == 1])
            current_ratio = mutated_samples / total_samples if total_samples > 0 else 0

            # Calculate progress toward target
            target_progress = (total_samples / self.dataset_size * 100) if self.dataset_size > 0 else 0
            logging.info(f"Iteration {iteration}: {total_samples}/{self.dataset_size} rows ({target_progress:.1f}% of target)")
            logging.info(f"  Mutation ratio: {current_ratio:.3f}, Target: {target_ratio:.3f}")

            # Early stopping conditions
            if total_samples >= self.dataset_size * early_stopping_threshold:
                logging.info(f"Early stopping: Reached {early_stopping_threshold*100:.0f}% of target size")
                break

            # Check for improvement
            if total_samples <= best_size:
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    logging.info("Early stopping: No improvement for 3 iterations")
                    logging.info(f"  Current: {total_samples} rows, Best: {best_size} rows")
                    break
            else:
                best_size = total_samples
                no_improvement_count = 0

            # Check if target dataset size is achieved
            if total_samples >= self.dataset_size:
                logging.info(f"✓ Target dataset size {self.dataset_size} achieved! (Actual: {total_samples})")
                break

            # Calculate how many more samples we need
            needed_samples = self.dataset_size - total_samples
            if needed_samples > 0:
                # Adjust duplication factor based on observed duplicate rates
                avg_duplicate_rate = sum(duplicate_rates) / len(duplicate_rates)
                if avg_duplicate_rate > 0.8:  # If >80% duplicates, increase factor
                    self.duplication_factor = min(50, int(self.duplication_factor * 1.5))
                    logging.info(f"High duplicate rate ({avg_duplicate_rate:.1%}), increasing duplication factor to {self.duplication_factor}")

                # Create additional samples
                # Set current_dataset_size to the number of additional rows needed
                # This will be properly converted to sequence samples in _create_sequence_samples
                self.current_dataset_size = needed_samples
                logging.info(f"Creating additional samples to reach target: need {needed_samples} more rows")
                additional_df = self._create_dataset()

                # Combine datasets and remove duplicates
                dataset_df = pd.concat([dataset_df, additional_df], ignore_index=True)
                dataset_df, iteration_duplicate_rate = self._remove_duplicates(dataset_df)
                duplicate_rates.append(iteration_duplicate_rate)

        # Save final dataset
        self._save_dataset(dataset_df)
        
        # Final statistics
        total_samples = len(dataset_df)
        mutated_samples = len(dataset_df[dataset_df['y'] == 1])
        final_ratio = mutated_samples / total_samples if total_samples > 0 else 0
        avg_duplicate_rate = sum(duplicate_rates) / len(duplicate_rates) if duplicate_rates else 0
        target_achieved_percent = (total_samples / self.dataset_size * 100) if self.dataset_size > 0 else 0

        logging.info(f"=== FINAL DATASET STATISTICS (REFILLING MODE) ===")
        logging.info(f"Target size: {self.dataset_size} rows (after deduplication)")
        logging.info(f"Actual size: {total_samples} rows ({target_achieved_percent:.1f}% of target)")
        logging.info(f"Mutation ratio: {final_ratio:.3f} ({mutated_samples}/{total_samples})")
        logging.info(f"Average duplicate rate: {avg_duplicate_rate:.1%}")
        logging.info(f"Final duplication factor: {self.duplication_factor}")

        if total_samples >= self.dataset_size * 0.95:
            logging.info(f"✓ Successfully achieved target dataset size!")
        else:
            logging.warning(f"⚠ Only achieved {target_achieved_percent:.1f}% of target size")

        # Clear checkpoint after successful completion
        if self.checkpoint_manager:
            self.checkpoint_manager.clear_checkpoint()
    
    def _create_dataset(self) -> pd.DataFrame:
        """Create the main dataset with period tracking."""
        # Load linked centroids
        centroids_df = self._load_linked_centroids()

        # Create sliding windows
        windows = self._create_sliding_windows(centroids_df)

        # Create sequence samples with metadata
        sequence_samples, window_metadata = self._create_sequence_samples(centroids_df, windows)

        # Process samples into final dataset format with period tracking
        dataset_df = self._process_samples_to_dataset(sequence_samples, window_metadata)

        return dataset_df
    
    def _load_linked_centroids(self) -> pd.DataFrame:
        """Load linked cluster centroids."""
        if not Path(self.linked_centroids_file).exists():
            raise FileNotFoundError(f"Linked centroids file not found: {self.linked_centroids_file}")
        
        logging.info(f"Loading linked centroids from: {self.linked_centroids_file}")
        return pd.read_csv(self.linked_centroids_file)
    
    def _create_sliding_windows(self, centroids_df: pd.DataFrame) -> List[Dict]:
        """Create sliding windows from consecutive periods."""
        periods = centroids_df['period'].unique()
        sorted_periods = natsorted(periods)
        
        logging.info(f"Creating sliding windows from {len(sorted_periods)} periods")
        
        # Adjust window size if needed
        actual_window_size = self.window_size
        if len(sorted_periods) < actual_window_size + 1:  # +1 for y period
            logging.warning(f"Not enough periods ({len(sorted_periods)}) for window size {self.window_size}")
            actual_window_size = len(sorted_periods) - 1
            logging.warning(f"Adjusted window size to {actual_window_size}")
        
        # Create windows
        windows = []
        num_windows = len(sorted_periods) - actual_window_size
        
        for i in range(num_windows):
            window = {
                'x': sorted_periods[i:i + actual_window_size],
                'y': sorted_periods[i + actual_window_size]
            }
            windows.append(window)
        
        logging.info(f"Created {len(windows)} sliding windows")
        return windows
    
    def _create_sequence_samples(self, centroids_df: pd.DataFrame, windows: List[Dict]) -> Tuple[List[List[str]], List[Dict]]:
        """Create sequence samples using parallel processing and checkpointing, with period tracking."""
        # Calculate number of sequence samples needed
        epitopes_pos_num = len(set(self.epitopes_positions))
        total_dataset_size = getattr(self, 'current_dataset_size', self.dataset_size)

        # dataset_size is the target AFTER deduplication
        # We need to overproduce to account for duplicates that will be removed
        target_final_rows = total_dataset_size  # This is our target after dedup
        target_sequence_samples = target_final_rows // epitopes_pos_num

        # Get expected duplicate rate for better initial estimation
        expected_dup_rate = self.dataset_config.get('expected_duplicate_rate', 0.8)

        # Calculate effective duplication factor based on expected duplicate rate
        if expected_dup_rate > 0 and expected_dup_rate < 1:
            # If we expect 80% duplicates, we need 5x samples (1 / 0.2 = 5)
            effective_duplication_factor = max(self.duplication_factor, 1 / (1 - expected_dup_rate))
        else:
            effective_duplication_factor = self.duplication_factor

        # Always apply duplication factor to reach target after dedup
        total_sequence_samples_needed = int(target_sequence_samples * effective_duplication_factor)

        # For very small datasets, ensure reasonable minimum overproduction
        if target_sequence_samples == 0 and target_final_rows > 0:
            # Special case: dataset_size is positive but smaller than epitope count
            # Create at least 1 sample per window to generate some data
            total_sequence_samples_needed = max(1, len(windows))
            logging.info(f"Very small dataset requested ({target_final_rows} rows), creating minimal samples")
        elif target_sequence_samples < 10 and total_sequence_samples_needed < target_sequence_samples * 5:
            total_sequence_samples_needed = min(target_sequence_samples * 10, 100)

        # Calculate samples per window
        samples_per_window = total_sequence_samples_needed // len(windows) if len(windows) > 0 else 0

        logging.info(f"=== SEQUENCE SAMPLE CREATION (Target: {total_dataset_size} final rows after dedup) ===")
        logging.info(f"Unique epitope positions: {epitopes_pos_num}")
        logging.info(f"Target FINAL dataset size (after dedup): {target_final_rows} rows")
        logging.info(f"Target sequence samples (final): {target_sequence_samples}")
        logging.info(f"Expected duplicate rate: {expected_dup_rate:.1%}")
        logging.info(f"Effective duplication factor: {effective_duplication_factor:.1f}")
        logging.info(f"Total samples to create (with overproduction): {total_sequence_samples_needed}")
        logging.info(f"Samples per window: {samples_per_window}")
        logging.info(f"Windows to process: {len(windows)}")
        logging.info(f"Expected rows before dedup: {total_sequence_samples_needed * epitopes_pos_num}")
        logging.info(f"Expected rows after dedup (estimate): {int(total_sequence_samples_needed * epitopes_pos_num * (1 - expected_dup_rate))}")

        # Early exit for very small datasets
        if samples_per_window == 0:
            logging.warning(f"Calculated 0 samples per window. Dataset size ({total_dataset_size}) too small for {len(windows)} windows.")
            logging.warning(f"Minimum dataset size for {len(windows)} windows with {epitopes_pos_num} epitope positions: {len(windows) * epitopes_pos_num}")
            return [], []  # Return empty lists for no processing

        # Create window metadata for period tracking
        window_metadata = []

        if self.parallel_windows and self.cache_enabled:
            samples, metadata = self._create_sequence_samples_parallel(
                centroids_df, windows, samples_per_window
            )
        else:
            logging.info("Using sequential processing (parallel disabled or cache unavailable)")
            samples, metadata = self._create_sequence_samples_sequential(
                centroids_df, windows, samples_per_window
            )

        return samples, metadata

    def _create_sequence_samples_parallel(self, centroids_df: pd.DataFrame,
                                         windows: List[Dict], samples_per_window: int) -> Tuple[List[List[str]], List[Dict]]:
        """Create samples using parallel window processing with checkpointing and period tracking."""
        start_time = time.time()

        # Check for existing checkpoint
        if self.checkpoint_manager:
            resume_point = self.checkpoint_manager.get_resume_point()
            if resume_point > 0:
                progress_info = self.checkpoint_manager.get_progress_info()
                logging.info(f"Resuming from checkpoint: window {resume_point}, "
                           f"already completed {progress_info['total_samples']} samples")
        else:
            resume_point = 0

        # Prepare data for workers (must be serializable)
        centroids_df_dict = centroids_df.to_dict()
        sequence_cache_data = self.sequence_cache.cache if self.sequence_cache else {}

        pipeline_config = {
            'epitopes_positions': self.epitopes_positions,
            'epitopes_similarity_threshold': self.epitopes_similarity_threshold,
            'sample_workers': self.sample_workers
        }

        # Create worker arguments for windows that need processing
        windows_to_process = [(i, w) for i, w in enumerate(windows) if i >= resume_point]
        worker_args = [
            (window_idx, window, samples_per_window, centroids_df_dict,
             sequence_cache_data, pipeline_config)
            for window_idx, window in windows_to_process
        ]

        logging.info(f"Processing {len(worker_args)} windows with {self.window_workers} parallel workers")

        # Process windows in parallel
        with multiprocessing.Pool(self.window_workers) as pool:
            try:
                # Process all windows in parallel
                results = pool.map(_process_window_worker, worker_args)

                # Save results with checkpointing
                if self.checkpoint_manager:
                    for window_idx, samples in results:
                        self.checkpoint_manager.save_window_results(window_idx, samples)

            except KeyboardInterrupt:
                logging.info("Interrupted by user. Saving checkpoint...")
                pool.terminate()
                pool.join()
                if self.checkpoint_manager:
                    logging.info("Checkpoint saved. You can resume later.")
                raise
            except Exception as e:
                logging.error(f"Parallel processing failed: {e}")
                pool.terminate()
                pool.join()
                raise

        # Collect all completed samples (including from checkpoint)
        if self.checkpoint_manager:
            all_samples = self.checkpoint_manager.get_completed_samples()
        else:
            all_samples = []
            for _, samples in results:
                all_samples.extend(samples)

        # Create metadata for all samples with period information
        all_metadata = []
        for window_idx, window in enumerate(windows):
            # Each window creates samples_per_window samples
            # Record the period range for each sample
            for _ in range(min(samples_per_window, len([s for _, samples in results if _ == window_idx for s in samples]))):
                all_metadata.append({
                    'period_start': window['x'][0] if window['x'] else '',
                    'period_end': window['y']
                })

        elapsed_time = time.time() - start_time
        failed_samples = (len(windows) * samples_per_window) - len(all_samples)

        logging.info(f"=== PARALLEL PROCESSING COMPLETED ===")
        logging.info(f"Total samples created: {len(all_samples)}")
        logging.info(f"Failed samples: {failed_samples}")
        logging.info(f"Processing time: {elapsed_time:.1f}s")
        logging.info(f"Throughput: {len(all_samples) / elapsed_time:.1f} samples/sec")

        return all_samples, all_metadata

    def _create_sequence_samples_sequential(self, centroids_df: pd.DataFrame,
                                          windows: List[Dict], samples_per_window: int) -> Tuple[List[List[str]], List[Dict]]:
        """Fallback sequential processing method with period tracking."""
        sequence_samples = []
        sequence_metadata = []
        failed_samples = 0

        for window_idx, window in enumerate(windows):
            logging.info(f"Processing window {window_idx + 1}/{len(windows)}: {window}")

            for sample_idx in range(samples_per_window):
                try:
                    if self.cache_enabled:
                        sample = self._create_single_sample_cached_fallback(centroids_df, window)
                    else:
                        sample = self._create_single_sample(centroids_df, window)
                    sequence_samples.append(sample)
                    # Track period metadata
                    sequence_metadata.append({
                        'period_start': window['x'][0] if window['x'] else '',
                        'period_end': window['y']
                    })
                except Exception as e:
                    failed_samples += 1
                    logging.warning(f"Failed to create sample {sample_idx + 1}: {e}")
                    continue

        logging.info(f"Sequential processing completed: {len(sequence_samples)} samples, {failed_samples} failed")
        return sequence_samples, sequence_metadata

    def _create_single_sample_cached_fallback(self, centroids_df: pd.DataFrame, window: Dict) -> List[str]:
        """Create a single sample using cache (fallback for sequential mode)."""
        pipeline_config = {
            'epitopes_positions': self.epitopes_positions,
            'epitopes_similarity_threshold': self.epitopes_similarity_threshold,
        }

        return _create_single_sample_cached(
            centroids_df, window, self.sequence_cache.cache, pipeline_config
        )
    
    def _create_single_sample(self, centroids_df: pd.DataFrame, window: Dict) -> List[str]:
        """Create a single sequence sample by linking clusters."""
        max_retries = 10
        
        for retry in range(max_retries):
            try:
                # Choose first sequence randomly
                first_sequence_info = self._choose_first_sequence(centroids_df, window)
                
                # Link remaining sequences
                sample_sequences = self._link_remaining_sequences(
                    centroids_df, first_sequence_info, window
                )
                
                return sample_sequences
                
            except SequencesMutatedTooMuch:
                if retry == max_retries - 1:
                    raise
                continue
        
        raise DatasetCreationError(f"Failed to create sample after {max_retries} retries")
    
    def _choose_first_sequence(self, centroids_df: pd.DataFrame, window: Dict) -> Dict:
        """Choose the first sequence randomly from the first period."""
        first_period = window['x'][0]
        first_period_rows = centroids_df[centroids_df['period'] == first_period]
        
        if first_period_rows.empty:
            raise DatasetCreationError(f"No clusters found for period: {first_period}")
        
        # Sample random cluster
        chosen_row = first_period_rows.sample(n=1).iloc[0]
        current_cluster = chosen_row['cluster']
        next_clusters_str = chosen_row['next_cluster']
        
        # Parse next clusters
        if pd.isna(next_clusters_str) or next_clusters_str == '':
            next_clusters = []
        else:
            next_clusters = [int(x) for x in str(next_clusters_str).split('-') if x.strip()]
        
        # Get sequence
        sequence = self._get_sequence_from_cluster(first_period, current_cluster)
        
        return {
            'sequence': sequence,
            'next_clusters': next_clusters
        }
    
    def _link_remaining_sequences(
        self, 
        centroids_df: pd.DataFrame, 
        first_sequence_info: Dict, 
        window: Dict
    ) -> List[str]:
        """Link remaining sequences following cluster links."""
        sequences = [first_sequence_info['sequence']]
        current_next_clusters = first_sequence_info['next_clusters']
        
        # Process x periods (input sequence)
        for i in range(1, len(window['x'])):
            current_period = window['x'][i]
            
            if not current_next_clusters:
                raise DatasetCreationError(f"No next clusters available for period: {current_period}")
            
            # Choose random cluster from available next clusters
            current_cluster = random.choice(current_next_clusters)
            
            # Get cluster info for current period
            cluster_row = centroids_df[
                (centroids_df['period'] == current_period) & 
                (centroids_df['cluster'] == current_cluster)
            ]
            
            if cluster_row.empty:
                raise DatasetCreationError(f"Cluster {current_cluster} not found in period {current_period}")
            
            # Get sequence and check mutation threshold
            sequence = self._get_sequence_from_cluster(current_period, current_cluster)
            previous_sequence = sequences[-1]
            
            # Retry if mutated too much
            max_mutation_retries = 10
            retry_count = 0
            
            while self._is_mutated_too_much(previous_sequence, sequence) and retry_count < max_mutation_retries:
                sequence = self._get_sequence_from_cluster(current_period, current_cluster)
                retry_count += 1
            
            if retry_count >= max_mutation_retries:
                raise SequencesMutatedTooMuch()
            
            sequences.append(sequence)
            
            # Update next clusters for next iteration
            next_clusters_str = cluster_row.iloc[0]['next_cluster']
            if pd.isna(next_clusters_str) or next_clusters_str == '':
                current_next_clusters = []
            else:
                current_next_clusters = [int(x) for x in str(next_clusters_str).split('-') if x.strip()]
        
        # Add y period sequence (target)
        y_period = window['y']
        if current_next_clusters:
            target_cluster = random.choice(current_next_clusters)
            target_sequence = self._get_sequence_from_cluster(y_period, target_cluster)
            sequences.append(target_sequence)
        else:
            raise DatasetCreationError(f"No clusters available for target period: {y_period}")
        
        return sequences
    
    def _get_sequence_from_cluster(self, period: str, cluster: int) -> str:
        """Get a random sequence from a specific cluster in a period."""
        period_file = f"{period}.csv"
        file_path = f"{self.periods_unique_dir}/{period_file}"
        
        if not Path(file_path).exists():
            raise DatasetCreationError(f"Period file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Get sequences from the cluster (no hardcoded length filter)
        cluster_sequences = df[df['cluster'] == cluster]
        
        if cluster_sequences.empty:
            raise DatasetCreationError(f"No sequences found for cluster {cluster} in period {period}")
        
        # Return random sequence
        chosen_sequence = cluster_sequences.sample(n=1).iloc[0]['sequence']
        return str(chosen_sequence)  # Ensure return type is str
    
    def _is_mutated_too_much(self, prev_sequence: str, current_sequence: str) -> bool:
        """Check if sequences are mutated beyond threshold in epitope regions."""
        mutated_epitopes = 0
        
        for pos in self.epitopes_positions:
            if pos < len(prev_sequence) and pos < len(current_sequence):
                if prev_sequence[pos] != current_sequence[pos]:
                    mutated_epitopes += 1
        
        return mutated_epitopes > self.max_similar_epitopes
    
    def _process_samples_to_dataset(self, sequence_samples: List[List[str]],
                                  window_metadata: List[Dict] = None) -> pd.DataFrame:
        """Process sequence samples into final dataset format with period tracking."""
        logging.info("Processing sequence samples into dataset format with period tracking...")

        # Handle empty sequence samples (for very small datasets)
        if not sequence_samples:
            logging.warning("No sequence samples to process. Returning empty dataset.")
            columns = ['y'] + [str(i) for i in range(self.window_size)] + ['period_start', 'period_end']
            return pd.DataFrame(columns=columns)

        # Check if streaming is enabled
        memory_config = self.config.get('optimization', {}).get('memory', {})
        use_streaming = memory_config.get('use_streaming', True)

        if use_streaming and len(sequence_samples) > self.batch_dataset_processor.batch_size:
            logging.info(f"Using streaming dataset creation for {len(sequence_samples)} samples")
            return self._process_samples_streaming(sequence_samples, window_metadata)
        else:
            logging.info(f"Using legacy dataset creation for {len(sequence_samples)} samples")
            return self._process_samples_legacy(sequence_samples, window_metadata)
    
    def _process_samples_streaming(self, sequence_samples: List[List[str]],
                                 window_metadata: List[Dict] = None) -> pd.DataFrame:
        """Process samples using streaming/batch approach with period tracking."""
        # Setup streaming output
        temp_output = f"{self.final_dataset_file}.temp"
        self.batch_dataset_processor.setup_output_file(temp_output)

        # Set metadata for period tracking
        if window_metadata:
            self.batch_dataset_processor.set_metadata(window_metadata)
            logging.info(f"Set metadata for {len(window_metadata)} samples in batch processor")

        # Process in batches
        self.batch_dataset_processor.process_in_batches(
            sequence_samples,
            description="dataset creation from sequence samples"
        )

        # Read back the streaming result
        if os.path.exists(temp_output):
            logging.info("Reading streaming dataset result...")
            dataset_df = pd.read_csv(temp_output)

            # Clean up temp file
            os.remove(temp_output)

            logging.info(f"Streaming dataset creation completed: {len(dataset_df)} rows")
            return dataset_df
        else:
            logging.warning("No streaming output generated, falling back to legacy method")
            return self._process_samples_legacy(sequence_samples, window_metadata)
    
    def _process_samples_legacy(self, sequence_samples: List[List[str]],
                              window_metadata: List[Dict] = None) -> pd.DataFrame:
        """Legacy method for processing samples with period tracking."""
        # Extract epitopes with context
        epitope_samples = self._extract_epitopes_with_context(sequence_samples)

        # Transform to ProtVec indices
        protvec_samples = self._transform_to_protvec_indices(epitope_samples)

        # Create final dataset with period metadata
        dataset_df = self._create_final_dataframe(protvec_samples, window_metadata)

        return dataset_df
    
    def _extract_epitopes_with_context(self, sequence_samples: List[List[str]]) -> List[List[List[str]]]:
        """Extract epitope regions with context from sequences."""
        epitope_samples = []
        
        for sample in sequence_samples:
            sample_epitopes = []
            for sequence in sample:
                # Remove asterisk if present
                if sequence.endswith('*'):
                    sequence = sequence[:-1]
                
                sequence_epitopes = []
                for position in self.epitopes_positions:
                    start_pos = max(0, position - self.context_size)
                    end_pos = min(len(sequence), position + self.context_size + 1)
                    epitope_context = sequence[start_pos:end_pos]
                    sequence_epitopes.append(epitope_context)
                
                sample_epitopes.append(sequence_epitopes)
            epitope_samples.append(sample_epitopes)
        
        return epitope_samples
    
    def _transform_to_protvec_indices(self, epitope_samples: List[List[List[str]]]) -> List[List[List[List[int]]]]:
        """Transform epitope contexts to ProtVec indices using parallel processing."""
        if not epitope_samples:
            return []

        # Check if parallel processing should be used
        use_parallel = (
            self.parallel_windows and
            len(epitope_samples) > 100 and
            self.window_workers > 1
        )

        if use_parallel:
            return self._transform_to_protvec_indices_parallel(epitope_samples)
        else:
            return self._transform_to_protvec_indices_sequential(epitope_samples)

    def _transform_to_protvec_indices_parallel(self, epitope_samples: List[List[List[str]]]) -> List[List[List[List[int]]]]:
        """Parallel ProtVec transformation."""
        start_time = time.time()
        logging.info(f"=== PARALLEL PROTVEC TRANSFORMATION ===")
        logging.info(f"Processing {len(epitope_samples)} samples with {self.window_workers} workers")

        # Determine chunk size for workers
        n_workers = min(self.window_workers, len(epitope_samples))
        chunk_size = max(1, len(epitope_samples) // n_workers)

        # Split samples into chunks
        chunks = []
        for i in range(0, len(epitope_samples), chunk_size):
            chunk = epitope_samples[i:i + chunk_size]
            chunks.append(chunk)

        # Prepare worker arguments
        worker_args = [
            (chunk, self.triplet_to_index, self.epitopes_positions, self.context_size, self.window_size)
            for chunk in chunks
        ]

        # Process chunks in parallel
        with multiprocessing.Pool(n_workers) as pool:
            try:
                chunk_results = pool.map(_transform_protvec_chunk_worker, worker_args)
            except Exception as e:
                logging.error(f"Parallel ProtVec transformation failed: {e}")
                pool.terminate()
                pool.join()
                # Fallback to sequential
                return self._transform_to_protvec_indices_sequential(epitope_samples)

        # Combine results from all chunks
        protvec_samples = []
        for chunk_result in chunk_results:
            protvec_samples.extend(chunk_result)

        elapsed_time = time.time() - start_time
        logging.info(f"Parallel ProtVec transformation completed in {elapsed_time:.1f}s")
        logging.info(f"Throughput: {len(protvec_samples) / elapsed_time:.1f} samples/sec")

        return protvec_samples

    def _transform_to_protvec_indices_sequential(self, epitope_samples: List[List[List[str]]]) -> List[List[List[List[int]]]]:
        """Sequential ProtVec transformation (fallback)."""
        protvec_samples = []

        for sample_idx, sample in enumerate(epitope_samples):
            if sample_idx % 100 == 0:
                logging.info(f"Processing ProtVec transformation for sample {sample_idx + 1}/{len(epitope_samples)}")

            sample_protvec = []
            for sequence_epitopes in sample:
                sequence_protvec = []
                for epitope_context in sequence_epitopes:
                    # Create triplets from epitope context
                    triplets = self._create_triplets_from_epitope(epitope_context)
                    # Convert triplets to ProtVec indices
                    triplet_indices = self._triplets_to_indices(triplets)
                    sequence_protvec.append(triplet_indices)
                sample_protvec.append(sequence_protvec)
            protvec_samples.append(sample_protvec)

        return protvec_samples
    
    def _create_triplets_from_epitope(self, epitope_context: str) -> List[str]:
        """Create 3-grams from epitope context."""
        sites_per_position = 1 + (2 * self.context_size)
        triplets_num = sites_per_position - 2
        
        if len(epitope_context) < 3:
            return []
        
        triplets = [epitope_context[i:i+3] for i in range(min(triplets_num, len(epitope_context) - 2))]
        return triplets
    
    def _triplets_to_indices(self, triplets: List[str]) -> List[int]:
        """Convert triplets to ProtVec indices using fast dictionary lookup."""
        indices = []
        for triplet in triplets:
            if triplet in self.triplet_to_index:
                indices.append(self.triplet_to_index[triplet])
            else:
                # Use a default index for unknown triplets
                indices.append(0)
        return indices
    
    def _create_final_dataframe(self, protvec_samples: List[List[List[List[int]]]],
                                window_metadata: List[Dict] = None) -> pd.DataFrame:
        """Create final dataset DataFrame using efficient batch creation with period tracking."""
        logging.info("Creating final dataset DataFrame with period tracking...")
        logging.info(f"Number of ProtVec samples: {len(protvec_samples)}")

        if not protvec_samples:
            columns = ['y'] + [str(i) for i in range(self.window_size)] + ['period_start', 'period_end']
            return pd.DataFrame(columns=columns)

        epitopes_count = len(protvec_samples[0][0]) if protvec_samples[0] else 0
        total_expected_rows = len(protvec_samples) * epitopes_count
        logging.info(f"Epitopes per sample: {epitopes_count}")
        logging.info(f"Expected final rows: {len(protvec_samples)} samples × {epitopes_count} epitopes = {total_expected_rows}")

        # Pre-allocate list for all rows (much faster than df.loc)
        logging.info("Starting row generation with period tracking...")
        all_rows = []

        for sample_idx, sample in enumerate(protvec_samples):
            if sample_idx % 50 == 0:
                logging.info(f"Processing sample {sample_idx + 1}/{len(protvec_samples)} for DataFrame creation")

            epitopes_count = len(sample[0])  # Number of epitope positions

            # Get period metadata for this sample if available
            if window_metadata and sample_idx < len(window_metadata):
                period_info = window_metadata[sample_idx]
                period_start = period_info.get('period_start', '')
                period_end = period_info.get('period_end', '')
            else:
                period_start = ''
                period_end = ''

            for epitope_idx in range(epitopes_count):
                row_data = []

                # Collect data for this epitope position across all sequences in the sample
                for sequence_idx in range(len(sample)):
                    if epitope_idx < len(sample[sequence_idx]):
                        row_data.append(sample[sequence_idx][epitope_idx])
                    else:
                        row_data.append([])  # Empty if epitope not available

                # Create row for dataset with period information
                dataset_row = self._create_dataset_row(row_data)
                # Add period metadata
                dataset_row.extend([period_start, period_end])
                all_rows.append(dataset_row)

        # Create DataFrame from all rows at once (much faster)
        columns = ['y'] + [str(i) for i in range(self.window_size)] + ['period_start', 'period_end']
        logging.info(f"Creating DataFrame with {len(all_rows)} rows and period tracking...")
        df = pd.DataFrame(all_rows, columns=columns)
        logging.info(f"DataFrame created successfully with {len(df)} rows")

        # Shuffle the dataset (will be handled differently for chronological split)
        logging.info("Shuffling dataset...")
        df = shuffle(df, random_state=42)
        df.reset_index(drop=True, inplace=True)
        logging.info("Dataset shuffling completed")

        return df
    
    def _create_dataset_row(self, row_data: List[List[int]]) -> List:
        """Create a single dataset row with mutation label."""
        # Separate target (y) from input sequences (x)
        input_sequences = row_data[:-1]
        target_sequence = row_data[-1]
        
        # Create mutation label
        if len(input_sequences) > 0:
            last_input_sequence = input_sequences[-1]
            # Check if target is similar to last input (mutation detection)
            mutation_label = 0 if self._sequences_similar(last_input_sequence, target_sequence) else 1
        else:
            mutation_label = 0
        
        # Convert lists to strings for DataFrame compatibility
        input_sequences_str = [str(seq) for seq in input_sequences]
        
        # Create final row
        result_row = [mutation_label] + input_sequences_str
        return result_row
    
    def _sequences_similar(self, seq1: List[int], seq2: List[int]) -> bool:
        """Check if two sequences are similar (have 3 or more matching elements)."""
        if not seq1 or not seq2:
            return True
        
        matching_elements = len(set(seq1) & set(seq2))
        return matching_elements >= 3
    
    def _remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Remove duplicate rows from dataset and return duplicate rate."""
        logging.info(f"Starting duplicate removal on {len(df)} rows...")
        
        # Check if streaming duplicate removal should be used
        memory_config = self.config.get('optimization', {}).get('memory', {})
        use_streaming = memory_config.get('use_streaming', True)
        batch_size = memory_config.get('dataset_creation_batch_size', 100)
        
        if use_streaming and len(df) > batch_size * 10:
            logging.info("Using streaming duplicate removal for large dataset")
            return self._remove_duplicates_streaming(df)
        else:
            logging.info("Using standard duplicate removal")
            return self._remove_duplicates_standard(df)
    
    def _remove_duplicates_streaming(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Remove duplicates using streaming approach for large datasets."""
        # Define columns to check for duplicates (all input columns, not y)
        duplicate_columns = [str(i) for i in range(self.window_size)]
        
        original_count = len(df)
        
        # Use a temporary file for streaming deduplication
        temp_file = f"{self.final_dataset_file}.dedup_temp"
        seen_signatures = set()
        rows_written = 0
        
        try:
            # Process in chunks
            memory_config = self.config.get('optimization', {}).get('memory', {})
            chunk_size = memory_config.get('dataset_creation_batch_size', 100)
            
            with open(temp_file, 'w') as outfile:
                # Write header
                header_written = False
                
                for start_idx in range(0, len(df), chunk_size):
                    end_idx = min(start_idx + chunk_size, len(df))
                    chunk = df.iloc[start_idx:end_idx]
                    
                    # Filter duplicates in chunk
                    unique_chunk_rows = []
                    
                    for _, row in chunk.iterrows():
                        # Create signature from duplicate columns
                        signature = tuple(row[duplicate_columns].values)
                        
                        if signature not in seen_signatures:
                            seen_signatures.add(signature)
                            unique_chunk_rows.append(row)
                    
                    # Write unique rows from this chunk
                    if unique_chunk_rows:
                        chunk_unique = pd.DataFrame(unique_chunk_rows)
                        chunk_unique.to_csv(outfile, mode='a', header=not header_written, index=False)
                        header_written = True
                        rows_written += len(chunk_unique)
                    
                    # Memory cleanup
                    if start_idx % (chunk_size * 10) == 0:
                        logging.info(f"Processed {end_idx}/{len(df)} rows for deduplication")
            
            # Read back the deduplicated data
            df_clean = pd.read_csv(temp_file)
            
            # Clean up temp file
            os.remove(temp_file)
            
            removed_count = original_count - len(df_clean)
            duplicate_rate = removed_count / original_count if original_count > 0 else 0
            
            logging.info(f"Streaming duplicate removal completed: {removed_count} duplicates removed ({duplicate_rate:.1%} duplication rate)")
            logging.info(f"Final dataset size after deduplication: {len(df_clean)} rows")
            
            return df_clean, duplicate_rate
            
        except Exception as e:
            # Clean up temp file in case of error
            if os.path.exists(temp_file):
                os.remove(temp_file)
            logging.error(f"Streaming duplicate removal failed: {e}")
            # Fallback to standard method
            return self._remove_duplicates_standard(df)
    
    def _remove_duplicates_standard(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Standard duplicate removal method."""
        # Define columns to check for duplicates (all input columns, not y)
        duplicate_columns = [str(i) for i in range(self.window_size)]
        
        original_count = len(df)
        logging.info("Identifying duplicates...")
        df_clean = df.drop_duplicates(subset=duplicate_columns)
        logging.info("Resetting index...")
        df_clean.reset_index(drop=True, inplace=True)
        
        removed_count = original_count - len(df_clean)
        duplicate_rate = removed_count / original_count if original_count > 0 else 0
        
        logging.info(f"Duplicate removal completed: {removed_count} duplicates removed ({duplicate_rate:.1%} duplication rate)")
        logging.info(f"Final dataset size after deduplication: {len(df_clean)} rows")
        
        return df_clean, duplicate_rate
    
    def _save_dataset(self, df: pd.DataFrame) -> None:
        """Save dataset to CSV file."""
        logging.info(f"Saving dataset to: {self.final_dataset_file}")
        df.to_csv(self.final_dataset_file, index=False)
        logging.info(f"Dataset saved: {len(df)} samples")


def run(config: Dict) -> None:
    """Main entry point for the dataset creation pipeline."""
    pipeline = DatasetCreationPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    # For testing purposes
    import sys
    sys.path.append('..')
    from scripts.config import load_config
    
    utils.setup_logger(verbose=True)
    config = load_config('../../configs/sars_cov_2_default.yaml')
    run(config)