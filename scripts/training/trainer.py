#!/usr/bin/env python3
"""
Model Training Module

This module handles the training loop with comprehensive metrics tracking:
- Config-driven hyperparameters (no hardcoded values)
- Full metrics: accuracy, precision, recall, F-score, MCC
- Model selection based on best MCC
- Training history tracking
"""

import logging
import time
import math
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.training.evaluator import ModelEvaluator
import scripts.utils as utils


class ModelTrainer:
    """Handles model training with comprehensive evaluation metrics."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trainer with configuration."""
        self.config = config
        
        # Extract hyperparameters from config - NO defaults allowed
        if 'learning_rate' not in config:
            raise ValueError("learning_rate must be specified in config")
        if 'batch_size' not in config:
            raise ValueError("batch_size must be specified in config") 
        if 'epochs' not in config:
            raise ValueError("epochs must be specified in config")
            
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        
        # Optional parameters
        self.print_interval = config.get('print_interval', 10)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def train(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train the model with comprehensive metrics tracking."""
        logging.info("Starting model training with comprehensive metrics...")
        logging.info(f"Training samples: {X_train.shape[1]}, Validation samples: {y_val.shape[0]}")
        logging.info(f"Epochs: {self.epochs}, Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Calculate batching parameters
        num_examples = X_train.shape[1]
        num_batches = math.floor(num_examples / self.batch_size)

        # Defensive check: ensure we have enough data for at least one batch
        if num_batches == 0:
            raise ValueError(
                f"Insufficient training data: {num_examples} samples < batch size {self.batch_size}. "
                f"Either reduce batch_size or provide more training data."
            )
        
        # Training history - comprehensive metrics tracking
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_fscore': [],
            'train_mcc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_fscore': [],
            'val_mcc': []
        }
        
        best_mcc = 0.0
        best_model_state = None
        
        # Find a validation batch with mutations for attention plotting
        plot_batch_size = X_train.shape[0]
        plot_batch_idx = self._find_mutation_batch(y_val, plot_batch_size)
        X_plot_batch = X_val[:, plot_batch_idx:plot_batch_idx + plot_batch_size, :]
        y_plot_batch = y_val[plot_batch_idx:plot_batch_idx + plot_batch_size]
        plot_batch_scores = []
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            train_metrics = self._train_epoch(model, X_train, y_train, criterion, optimizer, num_batches)
            
            # Validation phase  
            val_metrics, plot_scores = self._validate_epoch(model, X_val, y_val, X_plot_batch, y_plot_batch, criterion)
            
            # Update history
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_acc'].append(train_metrics['accuracy'])
            training_history['train_precision'].append(train_metrics['precision'])
            training_history['train_recall'].append(train_metrics['recall'])
            training_history['train_fscore'].append(train_metrics['fscore'])
            training_history['train_mcc'].append(train_metrics['mcc'])
            
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_acc'].append(val_metrics['accuracy'])
            training_history['val_precision'].append(val_metrics['precision'])
            training_history['val_recall'].append(val_metrics['recall'])
            training_history['val_fscore'].append(val_metrics['fscore'])
            training_history['val_mcc'].append(val_metrics['mcc'])
            
            plot_batch_scores.append(plot_scores)
            
            # Check for improvement based on MCC (as in original code)
            if val_metrics['mcc'] > best_mcc:
                best_mcc = val_metrics['mcc']
                best_model_state = model.state_dict().copy()
                elapsed_time = time.time() - start_time
                logging.info(f'Epoch {epoch + 1} Time {utils.get_time_string(elapsed_time)}')
                logging.info(f'Best MCC updated: V_loss {val_metrics["loss"]:.3f}\t'
                           f'V_acc {val_metrics["accuracy"]:.3f}\t'
                           f'V_pre {val_metrics["precision"]:.3f}\t'
                           f'V_rec {val_metrics["recall"]:.3f}\t'
                           f'V_fscore {val_metrics["fscore"]:.3f}\t'
                           f'V_mcc {val_metrics["mcc"]:.3f}')
            
            # Periodic logging
            if (epoch + 1) % self.print_interval == 0:
                elapsed_time = time.time() - start_time
                logging.info(f'Epoch {epoch + 1} Time {utils.get_time_string(elapsed_time)}')
                logging.info(f'T_loss {train_metrics["loss"]:.3f}\t'
                           f'T_acc {train_metrics["accuracy"]:.3f}\t'
                           f'T_pre {train_metrics["precision"]:.3f}\t'
                           f'T_rec {train_metrics["recall"]:.3f}\t'
                           f'T_fscore {train_metrics["fscore"]:.3f}\t'
                           f'T_mcc {train_metrics["mcc"]:.3f}')
                logging.info(f'V_loss {val_metrics["loss"]:.3f}\t'
                           f'V_acc {val_metrics["accuracy"]:.3f}\t'
                           f'V_pre {val_metrics["precision"]:.3f}\t'
                           f'V_rec {val_metrics["recall"]:.3f}\t'
                           f'V_fscore {val_metrics["fscore"]:.3f}\t'
                           f'V_mcc {val_metrics["mcc"]:.3f}')
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logging.info(f"Loaded best model with MCC: {best_mcc:.4f}")
        
        # Store plot data for visualization
        training_history['plot_batch_scores'] = plot_batch_scores
        training_history['plot_batch_labels'] = y_plot_batch
        
        return model, training_history
    
    def _train_epoch(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_batches: int
    ) -> Dict[str, float]:
        """Train for one epoch with comprehensive metrics."""
        model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Initialize hidden state
        hidden = model.init_hidden(self.batch_size)

        # Batch training loop
        for start_idx in range(0, X_train.shape[1] - self.batch_size + 1, self.batch_size):
            end_idx = start_idx + self.batch_size
            
            # Repackage hidden state to detach from history
            hidden = self._repackage_hidden(hidden)

            # Get batch
            X_batch = X_train[:, start_idx:end_idx, :]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass
            scores, _ = model(X_batch, hidden)
            loss = criterion(scores, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            predictions = self._predictions_from_output(scores)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy())

        # Calculate comprehensive metrics
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        accuracy, precision, recall, fscore, mcc = self.evaluator.evaluate(
            torch.tensor(all_labels), torch.tensor(all_predictions)
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'mcc': mcc
        }
    
    def _validate_epoch(
        self,
        model: nn.Module,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        X_plot_batch: torch.Tensor,
        y_plot_batch: torch.Tensor,
        criterion: nn.Module
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """Validate for one epoch with comprehensive metrics."""
        model.eval()
        
        with torch.no_grad():
            # Full validation
            hidden = model.init_hidden(y_val.shape[0])
            scores, _ = model(X_val, hidden)
            predictions = self._predictions_from_output(scores)
            predictions = predictions.view_as(y_val)
            
            # Calculate loss
            val_loss = criterion(scores, y_val).item()
            
            # Calculate comprehensive metrics
            accuracy, precision, recall, fscore, mcc = self.evaluator.evaluate(y_val, predictions)
            
            # Get plot batch scores for visualization
            plot_hidden = model.init_hidden(y_plot_batch.shape[0])
            plot_scores, _ = model(X_plot_batch, plot_hidden)
            
            val_metrics = {
                'loss': val_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'mcc': mcc
            }
            
            return val_metrics, plot_scores
    
    def _find_mutation_batch(self, y_val: torch.Tensor, batch_size: int) -> int:
        """Find a batch that contains at least one mutation for attention plotting."""
        for i in range(len(y_val) - batch_size + 1):
            if torch.sum(y_val[i:i + batch_size]) > 0:  # Contains at least one mutation
                return i
        return 0  # Fallback to first batch
    
    def _repackage_hidden(self, hidden):
        """Detach hidden states from their history."""
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(self._repackage_hidden(v) for v in hidden)
    
    def _predictions_from_output(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert logits to class predictions."""
        prob = F.softmax(scores, dim=1)
        _, predictions = prob.topk(1)
        return predictions
    
    def get_training_summary(self, history: Dict[str, List[float]]) -> Dict[str, float]:
        """Get a comprehensive summary of training metrics."""
        if not history['val_mcc']:
            return {}

        return {
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_mcc': max(history['val_mcc']),
            'final_train_mcc': history['train_mcc'][-1],
            'final_val_mcc': history['val_mcc'][-1],
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_fscore': max(history['val_fscore']),
            'final_val_fscore': history['val_fscore'][-1],
            'final_val_precision': history['val_precision'][-1],
            'final_val_recall': history['val_recall'][-1]
        }