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
        
        prot_vec_df = pd.read_csv(self.prot_vec_file)
        self.trigram_vecs = prot_vec_df.loc[:, prot_vec_df.columns != 'words'].values
        
        logging.info(f"Loaded ProtVec embeddings: {self.trigram_vecs.shape}")
    
    def load_and_split_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load dataset and split into train/validation/test sets."""
        logging.info("Loading and processing dataset...")
        
        # Load raw dataset
        X, y = self._load_dataset()
        
        # Split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_dataset(X, y)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process the main dataset."""
        if not Path(self.final_dataset_file).exists():
            raise FileNotFoundError(f"Dataset file not found: {self.final_dataset_file}")
        
        # Load main dataset
        df = pd.read_csv(self.final_dataset_file)
        logging.info(f"Loaded dataset with {len(df)} samples")
        
        # Extract labels
        labels = df['y'].values
        
        # Extract features (exclude 'y' column)
        feature_columns = [col for col in df.columns if col != 'y']
        feature_data = df[feature_columns].values
        
        # Process features
        X = self._process_features(feature_data)
        y = labels
        
        logging.info(f"Processed dataset shape: X={X.shape}, y={y.shape}")
        self._log_class_distribution(y)
        
        return X, y
    
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
    
    def _split_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split dataset into train/validation/test sets."""
        train_ratio, val_ratio, test_ratio = self.split_ratio
        
        logging.info(f"Splitting dataset: {train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}")
        
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