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
        return pd.read_csv(self.prot_vec_file)
    
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
        self._save_dataset(dataset_df)
        
        # Log statistics
        total_samples = len(dataset_df)
        mutated_samples = len(dataset_df[dataset_df['y'] == 1])
        mutation_ratio = mutated_samples / total_samples if total_samples > 0 else 0
        
        logging.info(f"Created dataset with {total_samples} samples")
        logging.info(f"Mutation ratio: {mutation_ratio:.3f} ({mutated_samples}/{total_samples})")
    
    def _run_with_refilling(self) -> None:
        """Run dataset creation with iterative refilling to achieve target ratio."""
        refiller_config = self.dataset_config['refiller']
        max_iterations = refiller_config.get('max_iterations', 20)
        target_ratio = refiller_config.get('target_mutated_ratio', 0.2)
        
        logging.info(f"Running dataset creation with refilling (target ratio: {target_ratio})")
        
        # Create initial dataset
        dataset_df = self._create_dataset()
        dataset_df = self._remove_duplicates(dataset_df)
        
        for iteration in range(1, max_iterations + 1):
            # Calculate current statistics
            total_samples = len(dataset_df)
            mutated_samples = len(dataset_df[dataset_df['y'] == 1])
            current_ratio = mutated_samples / total_samples if total_samples > 0 else 0
            
            logging.info(f"Iteration {iteration}/{max_iterations}: {total_samples} samples, "
                        f"mutation ratio: {current_ratio:.3f}")
            
            # Check if target ratio is achieved
            if current_ratio >= target_ratio:
                logging.info(f"Target mutation ratio {target_ratio} achieved!")
                break
            
            # Calculate how many more samples we need
            needed_samples = self.dataset_size - total_samples
            if needed_samples > 0:
                # Create additional samples
                self.current_dataset_size = needed_samples
                additional_df = self._create_dataset()
                
                # Combine datasets and remove duplicates
                dataset_df = pd.concat([dataset_df, additional_df], ignore_index=True)
                dataset_df = self._remove_duplicates(dataset_df)
        
        # Save final dataset
        self._save_dataset(dataset_df)
        
        # Final statistics
        total_samples = len(dataset_df)
        mutated_samples = len(dataset_df[dataset_df['y'] == 1])
        final_ratio = mutated_samples / total_samples if total_samples > 0 else 0
        
        logging.info(f"Final dataset: {total_samples} samples, "
                    f"mutation ratio: {final_ratio:.3f} ({mutated_samples}/{total_samples})")
    
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
        # Calculate number of samples needed
        epitopes_pos_num = len(set(self.epitopes_positions))  # Unique epitope positions
        total_dataset_size = getattr(self, 'current_dataset_size', self.dataset_size)
        samples_per_position = total_dataset_size // epitopes_pos_num
        samples_per_window = max(1, samples_per_position // len(windows))
        
        logging.info(f"Creating {samples_per_window} samples per window, {len(windows)} windows")
        
        sequence_samples = []
        
        for window_idx, window in enumerate(windows):
            logging.info(f"Processing window {window_idx + 1}/{len(windows)}: {window}")
            
            for sample_idx in range(samples_per_window):
                try:
                    sample = self._create_single_sample(centroids_df, window)
                    sequence_samples.append(sample)
                except Exception as e:
                    logging.warning(f"Failed to create sample {sample_idx + 1} for window {window_idx + 1}: {e}")
                    continue
        
        # Add remaining samples to last window if needed
        remaining_samples = samples_per_position - len(sequence_samples)
        if remaining_samples > 0 and windows:
            last_window = windows[-1]
            for i in range(remaining_samples):
                try:
                    sample = self._create_single_sample(centroids_df, last_window)
                    sequence_samples.append(sample)
                except Exception as e:
                    logging.warning(f"Failed to create remaining sample {i + 1}: {e}")
                    continue
        
        logging.info(f"Created {len(sequence_samples)} sequence samples")
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
        """Process sequence samples into final dataset format."""
        logging.info("Processing sequence samples into dataset format...")
        
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
        """Convert triplets to ProtVec indices."""
        indices = []
        for triplet in triplets:
            matching_rows = self.prot_vec_df[self.prot_vec_df['words'] == triplet]
            if not matching_rows.empty:
                indices.append(matching_rows.index[0])
            else:
                # Use a default index for unknown triplets
                indices.append(0)
        return indices
    
    def _create_final_dataframe(self, protvec_samples: List[List[List[List[int]]]]) -> pd.DataFrame:
        """Create final dataset DataFrame."""
        logging.info("Creating final dataset DataFrame...")
        
        # Create column headers
        columns = ['y'] + [str(i) for i in range(self.window_size)]
        df = pd.DataFrame(columns=columns)
        
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
                df.loc[len(df)] = dataset_row
        
        # Shuffle the dataset
        df = shuffle(df, random_state=42)
        df.reset_index(drop=True, inplace=True)
        
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
        
        # Create final row
        result_row = [mutation_label] + input_sequences
        return result_row
    
    def _sequences_similar(self, seq1: List[int], seq2: List[int]) -> bool:
        """Check if two sequences are similar (have 3 or more matching elements)."""
        if not seq1 or not seq2:
            return True
        
        matching_elements = len(set(seq1) & set(seq2))
        return matching_elements >= 3
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from dataset."""
        # Define columns to check for duplicates (all input columns, not y)
        duplicate_columns = [str(i) for i in range(self.window_size)]
        
        original_count = len(df)
        df_clean = df.drop_duplicates(subset=duplicate_columns)
        df_clean.reset_index(drop=True, inplace=True)
        
        removed_count = original_count - len(df_clean)
        if removed_count > 0:
            logging.info(f"Removed {removed_count} duplicate rows")
        
        return df_clean
    
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