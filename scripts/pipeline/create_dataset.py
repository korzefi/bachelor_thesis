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

import scripts.utils as utils
from scripts.utils import BatchProcessor, DataFrameChunker


class BatchDatasetProcessor(BatchProcessor):
    """Batch processor for streaming dataset creation."""
    
    def __init__(self, config: Dict, epitopes_positions: List[int], window_size: int):
        memory_config = config.get('memory_optimization', {})
        batch_size = memory_config.get('dataset_creation_batch_size', 100)
        super().__init__(config, batch_size, "BatchDatasetProcessor")
        
        self.epitopes_positions = epitopes_positions
        self.window_size = window_size
        self.temp_dir = None
        self.output_file = None
        
        # ProtVec batch size for transformation
        self.protvec_batch_size = memory_config.get('protvec_batch_size', 50)
    
    def setup_output_file(self, output_path: str):
        """Setup output file for streaming results."""
        self.output_file = output_path
        self.temp_dir = f"{output_path}_temp"
        utils.create_dir(self.temp_dir)
        
        # Remove existing output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def process_batch(self, batch_samples: List[List[str]]) -> pd.DataFrame:
        """Process a batch of sequence samples into dataset format."""
        # Extract epitopes with context from batch
        epitope_samples = self._extract_epitopes_with_context_batch(batch_samples)
        
        # Transform to ProtVec indices in sub-batches
        protvec_samples = self._transform_to_protvec_indices_batch(epitope_samples)
        
        # Create final dataframe from batch
        dataset_df = self._create_dataframe_from_batch(protvec_samples)
        
        return dataset_df
    
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
    
    def _create_dataframe_from_batch(self, protvec_samples: List[List[List[List[int]]]]) -> pd.DataFrame:
        """Create DataFrame from batch of ProtVec samples."""
        if not protvec_samples:
            columns = ['y'] + [str(i) for i in range(self.window_size)]
            return pd.DataFrame(columns=columns)
        
        # Pre-allocate list for rows
        all_rows = []
        
        for sample in protvec_samples:
            epitopes_count = len(sample[0])  # Number of epitope positions
            
            for epitope_idx in range(epitopes_count):
                row_data = []
                
                # Collect data for this epitope position across all sequences in the sample
                for sequence_idx in range(len(sample)):
                    if epitope_idx < len(sample[sequence_idx]):
                        row_data.append(sample[sequence_idx][epitope_idx])
                    else:
                        row_data.append([])  # Empty if epitope not available
                
                # Create row for dataset
                dataset_row = self._create_dataset_row(row_data)
                all_rows.append(dataset_row)
        
        # Create DataFrame from all rows
        columns = ['y'] + [str(i) for i in range(self.window_size)]
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
        
        # File paths
        self.linked_centroids_file = self.data_config['linked_centroids_file']
        self.periods_unique_dir = self.data_config['periods_unique_dir']
        self.prot_vec_file = self.data_config['prot_vec_file']
        self.final_dataset_file = self.data_config['final_dataset_file']
        
        # Load ProtVec embeddings
        self.prot_vec_df = self._load_prot_vec_embeddings()
        
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
        dataset_df = self._create_dataset()
        dataset_df, duplicate_rate = self._remove_duplicates(dataset_df)
        self._save_dataset(dataset_df)
        
        # Log statistics
        total_samples = len(dataset_df)
        mutated_samples = len(dataset_df[dataset_df['y'] == 1])
        mutation_ratio = mutated_samples / total_samples if total_samples > 0 else 0
        
        logging.info(f"Created dataset with {total_samples} samples")
        logging.info(f"Mutation ratio: {mutation_ratio:.3f} ({mutated_samples}/{total_samples})")
        logging.info(f"Final duplicate rate: {duplicate_rate:.1%}")
    
    def _run_with_refilling(self) -> None:
        """Run dataset creation with iterative refilling and early stopping."""
        refiller_config = self.dataset_config['refiller']
        max_iterations = refiller_config.get('max_iterations', 20)
        target_ratio = refiller_config.get('target_mutated_ratio', 0.2)
        early_stopping_threshold = refiller_config.get('early_stopping_threshold', 0.95)

        logging.info(f"Running with refilling (target ratio: {target_ratio})")

        dataset_df = self._create_dataset()
        dataset_df, initial_duplicate_rate = self._remove_duplicates(dataset_df)

        duplicate_rates = [initial_duplicate_rate]
        no_improvement_count = 0
        best_size = len(dataset_df)

        for iteration in range(1, max_iterations + 1):
            total_samples = len(dataset_df)
            mutated_samples = len(dataset_df[dataset_df['y'] == 1])
            current_ratio = mutated_samples / total_samples if total_samples > 0 else 0

            logging.info(f"Iteration {iteration}: {total_samples} samples, ratio: {current_ratio:.3f}")

            # Early stopping conditions
            if total_samples >= self.dataset_size * early_stopping_threshold:
                logging.info(f"Early stopping: {early_stopping_threshold*100}% of target reached")
                break

            # Check for improvement
            if total_samples <= best_size:
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    logging.info("Early stopping: No improvement for 3 iterations")
                    break
            else:
                best_size = total_samples
                no_improvement_count = 0

            # Check if target dataset size is achieved
            if total_samples >= self.dataset_size:
                logging.info(f"Target dataset size {self.dataset_size} achieved!")
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
                self.current_dataset_size = needed_samples
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
        
        logging.info(f"Final dataset: {total_samples} samples, "
                    f"mutation ratio: {final_ratio:.3f} ({mutated_samples}/{total_samples})")
        logging.info(f"Average duplicate rate: {avg_duplicate_rate:.1%}")
        logging.info(f"Final duplication factor: {self.duplication_factor}")
    
    def _create_dataset(self) -> pd.DataFrame:
        """Create the main dataset."""
        # Load linked centroids
        centroids_df = self._load_linked_centroids()
        
        # Create sliding windows
        windows = self._create_sliding_windows(centroids_df)
        
        # Create sequence samples
        sequence_samples = self._create_sequence_samples(centroids_df, windows)
        
        # Process samples into final dataset format
        dataset_df = self._process_samples_to_dataset(sequence_samples)
        
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
    
    def _create_sequence_samples(self, centroids_df: pd.DataFrame, windows: List[Dict]) -> List[List[str]]:
        """Create sequence samples by linking clusters across windows."""
        # Calculate number of sequence samples needed
        # Each sequence sample creates one row per epitope position in final dataset
        epitopes_pos_num = len(set(self.epitopes_positions))  # Unique epitope positions
        total_dataset_size = getattr(self, 'current_dataset_size', self.dataset_size)
        
        # Total sequence samples needed = target dataset size / epitopes per sample
        # Apply duplication factor to compensate for duplicate removal
        base_sequence_samples_needed = total_dataset_size // epitopes_pos_num
        total_sequence_samples_needed = int(base_sequence_samples_needed * self.duplication_factor)
        samples_per_window = max(1, total_sequence_samples_needed // len(windows))
        
        logging.info(f"Unique epitope positions: {epitopes_pos_num}")
        logging.info(f"Target dataset size: {total_dataset_size}")
        logging.info(f"Base sequence samples needed: {base_sequence_samples_needed}")
        logging.info(f"Duplication factor: {self.duplication_factor}")
        logging.info(f"Total sequence samples needed (with duplication factor): {total_sequence_samples_needed}")
        logging.info(f"Creating {samples_per_window} samples per window, {len(windows)} windows")
        logging.info(f"Expected total sequence samples: {samples_per_window * len(windows)}")
        logging.info(f"Expected pre-deduplication rows: {samples_per_window * len(windows)} × {epitopes_pos_num} = {samples_per_window * len(windows) * epitopes_pos_num}")
        
        sequence_samples = []
        failed_samples = 0
        
        for window_idx, window in enumerate(windows):
            logging.info(f"Processing window {window_idx + 1}/{len(windows)}: {window}")
            
            for sample_idx in range(samples_per_window):
                try:
                    sample = self._create_single_sample(centroids_df, window)
                    sequence_samples.append(sample)
                except Exception as e:
                    failed_samples += 1
                    logging.warning(f"Failed to create sample {sample_idx + 1} for window {window_idx + 1}: {e}")
                    continue
        
        # Add remaining samples to last window if needed
        remaining_samples = total_sequence_samples_needed - len(sequence_samples)
        if remaining_samples > 0 and windows:
            logging.info(f"Creating {remaining_samples} additional samples in last window")
            last_window = windows[-1]
            for i in range(remaining_samples):
                try:
                    sample = self._create_single_sample(centroids_df, last_window)
                    sequence_samples.append(sample)
                except Exception as e:
                    failed_samples += 1
                    logging.warning(f"Failed to create remaining sample {i + 1}: {e}")
                    continue
        
        logging.info(f"Created {len(sequence_samples)} sequence samples")
        if failed_samples > 0:
            logging.warning(f"Failed to create {failed_samples} samples due to mutation threshold or other errors")
        
        expected_final_rows = len(sequence_samples) * epitopes_pos_num
        logging.info(f"Expected final dataset rows: {len(sequence_samples)} × {epitopes_pos_num} = {expected_final_rows}")
        
        return sequence_samples
    
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
    
    def _process_samples_to_dataset(self, sequence_samples: List[List[str]]) -> pd.DataFrame:
        """Process sequence samples into final dataset format using streaming."""
        logging.info("Processing sequence samples into dataset format...")
        
        # Check if streaming is enabled
        memory_config = self.config.get('memory_optimization', {})
        use_streaming = memory_config.get('use_streaming', True)
        
        if use_streaming and len(sequence_samples) > self.batch_dataset_processor.batch_size:
            logging.info(f"Using streaming dataset creation for {len(sequence_samples)} samples")
            return self._process_samples_streaming(sequence_samples)
        else:
            logging.info(f"Using legacy dataset creation for {len(sequence_samples)} samples")
            return self._process_samples_legacy(sequence_samples)
    
    def _process_samples_streaming(self, sequence_samples: List[List[str]]) -> pd.DataFrame:
        """Process samples using streaming/batch approach."""
        # Setup streaming output
        temp_output = f"{self.final_dataset_file}.temp"
        self.batch_dataset_processor.setup_output_file(temp_output)
        
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
            return self._process_samples_legacy(sequence_samples)
    
    def _process_samples_legacy(self, sequence_samples: List[List[str]]) -> pd.DataFrame:
        """Legacy method for processing samples (fallback for small datasets)."""
        # Extract epitopes with context
        epitope_samples = self._extract_epitopes_with_context(sequence_samples)
        
        # Transform to ProtVec indices
        protvec_samples = self._transform_to_protvec_indices(epitope_samples)
        
        # Create final dataset
        dataset_df = self._create_final_dataframe(protvec_samples)
        
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
        """Transform epitope contexts to ProtVec indices."""
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
    
    def _create_final_dataframe(self, protvec_samples: List[List[List[List[int]]]]) -> pd.DataFrame:
        """Create final dataset DataFrame using efficient batch creation."""
        logging.info("Creating final dataset DataFrame...")
        logging.info(f"Number of ProtVec samples: {len(protvec_samples)}")
        
        if not protvec_samples:
            columns = ['y'] + [str(i) for i in range(self.window_size)]
            return pd.DataFrame(columns=columns)
        
        epitopes_count = len(protvec_samples[0][0]) if protvec_samples[0] else 0
        total_expected_rows = len(protvec_samples) * epitopes_count
        logging.info(f"Epitopes per sample: {epitopes_count}")
        logging.info(f"Expected final rows: {len(protvec_samples)} samples × {epitopes_count} epitopes = {total_expected_rows}")
        
        # Pre-allocate list for all rows (much faster than df.loc)
        logging.info("Starting row generation...")
        all_rows = []
        
        for sample_idx, sample in enumerate(protvec_samples):
            if sample_idx % 50 == 0:
                logging.info(f"Processing sample {sample_idx + 1}/{len(protvec_samples)} for DataFrame creation")
            
            epitopes_count = len(sample[0])  # Number of epitope positions
            
            for epitope_idx in range(epitopes_count):
                row_data = []
                
                # Collect data for this epitope position across all sequences in the sample
                for sequence_idx in range(len(sample)):
                    if epitope_idx < len(sample[sequence_idx]):
                        row_data.append(sample[sequence_idx][epitope_idx])
                    else:
                        row_data.append([])  # Empty if epitope not available
                
                # Create row for dataset
                dataset_row = self._create_dataset_row(row_data)
                all_rows.append(dataset_row)
        
        # Create DataFrame from all rows at once (much faster)
        columns = ['y'] + [str(i) for i in range(self.window_size)]
        logging.info(f"Creating DataFrame with {len(all_rows)} rows...")
        df = pd.DataFrame(all_rows, columns=columns)
        logging.info(f"DataFrame created successfully with {len(df)} rows")
        
        # Shuffle the dataset
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
        memory_config = self.config.get('memory_optimization', {})
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
            memory_config = self.config.get('memory_optimization', {})
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