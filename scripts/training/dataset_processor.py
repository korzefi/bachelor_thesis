#!/usr/bin/env python3
"""
Dataset Processing Module

This module handles loading, preprocessing, and splitting of datasets for training:
- Loading dataset from CSV files
- Converting string representations to vectors
- Splitting into train/validation/test sets
- Converting to PyTorch tensors
"""

import logging
import ast
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class DatasetProcessor:
    """Handles dataset loading and preprocessing."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.data_config = config['data']
        self.train_config = config['train']
        
        # Paths
        self.final_dataset_file = self.data_config['final_dataset_file']
        self.prot_vec_file = self.data_config['prot_vec_file']
        
        # Split configuration
        self.split_ratio = self.train_config['split_ratio']
        
        # Load ProtVec embeddings once
        self._load_prot_vec_embeddings()
    
    def _load_prot_vec_embeddings(self) -> None:
        """Load ProtVec embeddings for sequence processing."""
        logging.info(f"Loading ProtVec embeddings from: {self.prot_vec_file}")
        
        if not Path(self.prot_vec_file).exists():
            raise FileNotFoundError(f"ProtVec file not found: {self.prot_vec_file}")
        
        prot_vec_df = pd.read_csv(self.prot_vec_file, sep='\t')
        self.trigram_vecs = prot_vec_df.loc[:, prot_vec_df.columns != 'words'].values
        
        logging.info(f"Loaded ProtVec embeddings: {self.trigram_vecs.shape}")
    
    def load_and_split_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load dataset and split into train/validation/test sets."""
        logging.info("Loading and processing dataset...")

        # Load raw dataset with period information
        X, y, period_start, period_end = self._load_dataset()

        # Split dataset (will use chronological or random based on config and data availability)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_dataset(X, y, period_start, period_end)

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and process the main dataset with period information."""
        if not Path(self.final_dataset_file).exists():
            raise FileNotFoundError(f"Dataset file not found: {self.final_dataset_file}")

        # Load main dataset
        df = pd.read_csv(self.final_dataset_file)
        logging.info(f"Loaded dataset with {len(df)} samples")

        # Check if period information is available
        has_period_info = 'period_start' in df.columns and 'period_end' in df.columns

        if has_period_info:
            logging.info("Period information found in dataset - will enable chronological splitting")
            period_start = df['period_start'].values
            period_end = df['period_end'].values
        else:
            logging.info("No period information in dataset - chronological splitting not available")
            period_start = None
            period_end = None

        # Extract labels
        labels = df['y'].values

        # Extract features (exclude 'y' and period columns)
        exclude_columns = ['y', 'period_start', 'period_end']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        feature_data = df[feature_columns].values

        # Process features
        X = self._process_features(feature_data)
        y = labels

        logging.info(f"Processed dataset shape: X={X.shape}, y={y.shape}")
        self._log_class_distribution(y)

        return X, y, period_start, period_end
    
    def _process_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Convert string representations of indices to summed ProtVec vectors."""
        logging.info("Processing feature data...")
        processed_features = []
        
        for row_idx, row in enumerate(feature_data):
            if row_idx % 1000 == 0:
                logging.info(f"Processing row {row_idx + 1}/{len(feature_data)}")
            
            processed_row = []
            for cell in row:
                processed_vector = self._process_single_cell(cell)
                processed_row.append(processed_vector)
            
            processed_features.append(processed_row)
        
        # Convert to numpy array and transpose to get shape: [batch_size, seq_length, feature_dim]
        X = np.array(processed_features)
        X = np.transpose(X, (0, 1, 2))
        
        return X
    
    def _process_single_cell(self, cell: Any) -> np.ndarray:
        """Process a single cell containing string representation of indices."""
        try:
            # Convert string representation of list to actual list
            if isinstance(cell, str) and cell.strip():
                indices = ast.literal_eval(cell)
            else:
                indices = cell
            
            # Convert indices to vectors and sum them
            if isinstance(indices, list) and len(indices) > 0:
                vectors = []
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(self.trigram_vecs):
                        vectors.append(self.trigram_vecs[idx])
                
                if vectors:
                    summed_vector = np.sum(vectors, axis=0)
                else:
                    summed_vector = np.zeros(100)  # 100-dimensional ProtVec
            else:
                summed_vector = np.zeros(100)
            
            return summed_vector
            
        except Exception as e:
            logging.warning(f"Error processing cell {cell}: {e}")
            return np.zeros(100)
    
    def _log_class_distribution(self, y: np.ndarray) -> None:
        """Log the class distribution of the dataset."""
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        logging.info(f"Class distribution: {class_dist}")
        
        # Calculate class balance
        total = len(y)
        for class_label, count in class_dist.items():
            percentage = (count / total) * 100
            logging.info(f"Class {class_label}: {count} samples ({percentage:.1f}%)")
    
    def _split_dataset(self, X: np.ndarray, y: np.ndarray,
                      period_start: np.ndarray = None, period_end: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split dataset into train/validation/test sets using configured strategy."""
        split_strategy = self.train_config.get('split_strategy', 'random')

        if split_strategy == 'chronological' and period_end is not None:
            logging.info("Using CHRONOLOGICAL splitting to prevent data leakage")
            return self._chronological_split(X, y, period_start, period_end)
        else:
            if split_strategy == 'chronological':
                logging.warning("Chronological split requested but period information not available. Falling back to random split.")
            logging.info("Using RANDOM splitting")
            return self._random_split(X, y)

    def _random_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random split (original implementation)."""
        train_ratio, val_ratio, test_ratio = self.split_ratio

        logging.info(f"Random split ratios: {train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}")

        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_ratio + test_ratio),
            stratify=y,
            random_state=42
        )

        # Second split: val vs test
        if test_ratio > 0:
            relative_test_size = test_ratio / (val_ratio + test_ratio)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=relative_test_size,
                stratify=y_temp,
                random_state=42
            )
        else:
            X_val, X_test = X_temp, np.array([])
            y_val, y_test = y_temp, np.array([])

        logging.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32) if len(X_test) > 0 else torch.empty(0)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long) if len(y_test) > 0 else torch.empty(0)

        return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

    def _chronological_split(self, X: np.ndarray, y: np.ndarray,
                           period_start: np.ndarray, period_end: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chronological split to prevent data leakage in time series."""
        train_ratio, val_ratio, test_ratio = self.split_ratio
        chrono_config = self.train_config.get('chronological_split', {})
        method = chrono_config.get('method', 'percentage')
        verbose = chrono_config.get('verbose_logging', True)
        min_samples = chrono_config.get('min_samples_per_split', 100)

        logging.info(f"=== CHRONOLOGICAL SPLIT (method: {method}) ===")

        # Get unique periods and sort them chronologically
        unique_periods = sorted(set(period_end))
        num_periods = len(unique_periods)

        if verbose:
            logging.info(f"Total unique periods: {num_periods}")
            logging.info(f"Period range: {unique_periods[0]} to {unique_periods[-1]}")

        if method == 'percentage':
            # Calculate period boundaries based on percentages
            train_end_idx = int(num_periods * train_ratio)
            val_end_idx = int(num_periods * (train_ratio + val_ratio))

            train_periods = unique_periods[:train_end_idx]
            val_periods = unique_periods[train_end_idx:val_end_idx]
            test_periods = unique_periods[val_end_idx:]

            if verbose:
                logging.info(f"Train periods: {train_periods[0]} to {train_periods[-1] if train_periods else 'None'} ({len(train_periods)} periods)")
                logging.info(f"Val periods: {val_periods[0] if val_periods else 'None'} to {val_periods[-1] if val_periods else 'None'} ({len(val_periods)} periods)")
                logging.info(f"Test periods: {test_periods[0] if test_periods else 'None'} to {test_periods[-1] if test_periods else 'None'} ({len(test_periods)} periods)")

        else:  # method == 'fixed_periods'
            # Use fixed period ranges from config
            train_range = chrono_config.get('train_periods', [])
            val_range = chrono_config.get('val_periods', [])
            test_range = chrono_config.get('test_periods', [])

            if not (train_range and val_range and test_range):
                raise ValueError("Fixed period ranges must be specified in config for 'fixed_periods' method")

            # Filter periods within specified ranges
            train_periods = [p for p in unique_periods if train_range[0] <= p <= train_range[1]]
            val_periods = [p for p in unique_periods if val_range[0] <= p <= val_range[1]]
            test_periods = [p for p in unique_periods if test_range[0] <= p <= test_range[1]]

        # Validate no temporal overlap between splits
        self._validate_no_temporal_overlap(train_periods, val_periods, test_periods)

        # Create masks for each split based on period_end
        train_mask = np.isin(period_end, train_periods)
        val_mask = np.isin(period_end, val_periods)
        test_mask = np.isin(period_end, test_periods)

        # Apply masks to get splits
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Log split information
        logging.info(f"Chronological split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # Validate minimum samples
        if len(X_train) < min_samples:
            logging.warning(f"Training set has only {len(X_train)} samples (minimum: {min_samples})")
        if len(X_val) < min_samples:
            logging.warning(f"Validation set has only {len(X_val)} samples (minimum: {min_samples})")
        if len(X_test) < min_samples:
            logging.warning(f"Test set has only {len(X_test)} samples (minimum: {min_samples})")

        # Log class distribution for each split
        if verbose:
            self._log_split_class_distribution(y_train, y_val, y_test)

        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32) if len(X_test) > 0 else torch.empty(0)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long) if len(y_test) > 0 else torch.empty(0)

        return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

    def _validate_no_temporal_overlap(self, train_periods: list, val_periods: list, test_periods: list) -> None:
        """Validate that there is no temporal overlap between train, validation, and test sets."""
        # Check for overlap between train and validation
        train_val_overlap = set(train_periods) & set(val_periods)
        if train_val_overlap:
            raise ValueError(f"Temporal overlap detected between train and validation sets: {train_val_overlap}")

        # Check for overlap between train and test
        train_test_overlap = set(train_periods) & set(test_periods)
        if train_test_overlap:
            raise ValueError(f"Temporal overlap detected between train and test sets: {train_test_overlap}")

        # Check for overlap between validation and test
        val_test_overlap = set(val_periods) & set(test_periods)
        if val_test_overlap:
            raise ValueError(f"Temporal overlap detected between validation and test sets: {val_test_overlap}")

        # Validate chronological order
        if train_periods and val_periods:
            max_train = max(train_periods)
            min_val = min(val_periods)
            if max_train > min_val:
                logging.warning(f"Train periods ({max_train}) extend beyond validation periods ({min_val}). "
                              "This may indicate data leakage.")

        if val_periods and test_periods:
            max_val = max(val_periods)
            min_test = min(test_periods)
            if max_val > min_test:
                logging.warning(f"Validation periods ({max_val}) extend beyond test periods ({min_test}). "
                              "This may indicate data leakage.")

        logging.info("✓ No temporal overlap detected between splits - chronological integrity preserved")

    def _log_split_class_distribution(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> None:
        """Log class distribution for each split."""
        for split_name, split_data in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
            if len(split_data) > 0:
                unique, counts = np.unique(split_data, return_counts=True)
                class_dist = dict(zip(unique.tolist(), counts.tolist()))
                total = len(split_data)
                dist_str = ", ".join([f"Class {cls}: {cnt} ({cnt/total*100:.1f}%)" for cls, cnt in class_dist.items()])
                logging.info(f"{split_name} set class distribution: {dist_str}")
    
    def get_dataset_info(self, X: torch.Tensor) -> Dict[str, int]:
        """Get information about the dataset dimensions."""
        if len(X.shape) >= 3:
            batch_size, seq_length, input_dim = X.shape[:3]
        else:
            batch_size, seq_length, input_dim = X.shape[0], 1, X.shape[1] if len(X.shape) > 1 else 1
        
        return {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'input_dim': input_dim,
            'output_dim': 2  # Binary classification
        }