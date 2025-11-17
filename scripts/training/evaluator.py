#!/usr/bin/env python3
"""
Model Evaluation Module

This module provides comprehensive evaluation metrics for model assessment:
- Standard classification metrics (accuracy, precision, recall, F-score)
- Matthews Correlation Coefficient (MCC) - crucial for imbalanced datasets
- ROC curves and AUC
- Confusion matrices
- Detailed performance analysis
"""

import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, confusion_matrix, roc_curve, auc,
    classification_report
)

from sklearn.linear_model import LogisticRegression

class ModelEvaluator:
    """Comprehensive model evaluation with all metrics from original code."""

    def __init__(self, device: torch.device = None) -> None:
        """Initialize evaluator."""
        self.device = device if device is not None else torch.device('cpu')
    
    def evaluate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float, float, float, float]:
        """
        Evaluate performance metrics given true labels and predictions.
        
        Returns: (accuracy, precision, recall, f1score, mcc)
        This matches the original evaluation.evaluate() function signature.
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy().flatten()
        
        # Handle empty arrays
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Calculate metrics
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1score = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            logging.warning(f"Error in metrics calculation: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        return accuracy, precision, recall, f1score, mcc
    
    def comprehensive_evaluation(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation including ROC curves and detailed metrics.
        This replaces the original test_model() function with enhanced functionality.
        """
        logging.info("Performing comprehensive model evaluation...")

        # Move data to device
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        model.eval()

        with torch.no_grad():
            # Get predictions
            hidden = model.init_hidden(y_test.shape[0])
            test_scores, attention_weights = model(X_test, hidden)
            predictions = self._predictions_from_output(test_scores)
            predictions = predictions.view_as(y_test)
            
            # Get prediction probabilities
            pred_probs = F.softmax(test_scores, dim=1)
            
            # Calculate comprehensive metrics
            accuracy, precision, recall, fscore, mcc = self.evaluate(y_test, predictions)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_test.cpu().numpy(), predictions.cpu().numpy())
            
            # Calculate ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(
                y_test.cpu().numpy(), 
                pred_probs[:, 1].cpu().numpy()
            )
            roc_auc = auc(fpr, tpr)
            
            # Detailed classification report
            class_report = classification_report(
                y_test.cpu().numpy(), 
                predictions.cpu().numpy(),
                output_dict=True
            )
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'mcc': mcc,
                'confusion_matrix': conf_matrix.tolist(),
                'roc_auc': roc_auc,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'predictions': predictions.cpu().numpy().tolist(),
                'true_labels': y_test.cpu().numpy().tolist(),
                'probabilities': pred_probs.cpu().numpy().tolist(),
                'attention_weights': attention_weights.cpu().numpy().tolist() if attention_weights is not None else None,
                'classification_report': class_report
            }
            
            # Log results (matching original format)
            logging.info(f'Test_acc {accuracy:.3f}\t'
                        f'Test_pre {precision:.3f}\t'
                        f'Test_rec {recall:.3f}\t'
                        f'Test_fscore {fscore:.3f}\t'
                        f'Test_mcc {mcc:.3f}\t'
                        f'Test_auc {roc_auc:.3f}')
            
            return results
    
    def logistic_regression_baseline(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Compute logistic regression baseline as in original code.
        This matches the original logistic_regression() function.
        """

        logging.info("Computing logistic regression baseline...")
        
        # Reshape data to linear format (flatten temporal dimension)
        X_train_linear = self._reshape_to_linear(X_train, window_size)
        X_val_linear = self._reshape_to_linear(X_val, window_size)
        
        # Convert to numpy
        X_train_np = np.array(X_train_linear)
        X_val_np = np.array(X_val_linear)
        y_train_np = y_train.cpu().numpy()
        y_val_np = y_val.cpu().numpy()
        
        # Fit logistic regression
        clf = LogisticRegression(random_state=0, max_iter=1000)
        clf.fit(X_train_np, y_train_np)
        
        # Evaluate on training set
        train_pred = clf.predict(X_train_np)
        train_acc, train_pre, train_rec, train_fscore, train_mcc = self.evaluate(
            y_train_np, train_pred
        )
        
        # Evaluate on validation set
        val_pred = clf.predict(X_val_np)
        val_acc, val_pre, val_rec, val_fscore, val_mcc = self.evaluate(
            y_val_np, val_pred
        )
        
        # ROC curve
        y_pred_proba = clf.predict_proba(X_val_np)[:, 1]
        fpr, tpr, _ = roc_curve(y_val_np, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Log results (matching original format)
        logging.info('Logistic regression baseline:')
        logging.info(f'T_acc {train_acc:.3f}\t'
                    f'T_pre {train_pre:.3f}\t'
                    f'T_rec {train_rec:.3f}\t'
                    f'T_fscore {train_fscore:.3f}\t'
                    f'T_mcc {train_mcc:.3f}')
        logging.info(f'V_acc {val_acc:.3f}\t'
                    f'V_pre {val_pre:.3f}\t'
                    f'V_rec {val_rec:.3f}\t'
                    f'V_fscore {val_fscore:.3f}\t'
                    f'V_mcc {val_mcc:.3f}')
        logging.info(f'ROC AUC: {roc_auc:.3f}')
        
        return {
            'train_metrics': {
                'accuracy': train_acc,
                'precision': train_pre,
                'recall': train_rec,
                'fscore': train_fscore,
                'mcc': train_mcc
            },
            'val_metrics': {
                'accuracy': val_acc,
                'precision': val_pre,
                'recall': val_rec,
                'fscore': val_fscore,
                'mcc': val_mcc
            },
            'roc_auc': roc_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    
    def _predictions_from_output(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert logits to class predictions (from original code)."""
        prob = F.softmax(scores, dim=1)
        _, predictions = prob.topk(1)
        return predictions
    
    def _reshape_to_linear(self, X: torch.Tensor, window_size: int) -> List[List[float]]:
        """
        Reshape temporal data to linear format for logistic regression.
        This matches the original reshape_to_linear() function.
        """
        # X shape: [seq_length, batch_size, feature_dim]
        # We want: [batch_size, flattened_features]

        X_np = X.permute(1, 0, 2)
        X_np = X_np.cpu().numpy()
        batch_size, seq_length, feature_dim = X_np.shape

        # Take the last window_size timesteps
        start_idx = max(0, seq_length - window_size)
        X_windowed = X_np[:, start_idx:, :]  # [batch_size, window_size, feature_dim]

        # Flatten temporal and feature dimensions
        reshaped = []
        for batch_idx in range(batch_size):
            flattened = X_windowed[batch_idx, :, :].flatten().tolist()
            reshaped.append(flattened)

        return reshaped
    
    def list_summary(self, name: str, data: np.ndarray) -> None:
        """
        Print summary of unique values and their counts.
        This matches the original list_summary() function.
        """
        logging.info(f"{name}:")
        unique, count = np.unique(data, return_counts=True)
        summary = dict(zip(unique, count))
        logging.info(str(summary))