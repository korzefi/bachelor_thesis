#!/usr/bin/env python3
"""
Training Visualization Module

This module handles all visualization aspects of training:
- Training history plots (loss, accuracy, all metrics)
- Attention weight visualization
- ROC curves
- Confusion matrices
- Prediction dynamics plots
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class TrainingVisualizer:
    """Handles all training-related visualizations."""
    
    def __init__(self, results_dir: str) -> None:
        """Initialize visualizer with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('ggplot')
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        show_attention_dynamics: bool = True
    ) -> None:
        """Plot comprehensive training history with generic plotting."""
        logging.info("Creating training history plots...")
        
        # Define metrics to plot with their display names
        metrics_config = [
            ('loss', 'Loss', 'Loss'),
            ('acc', 'Accuracy', 'Accuracy'),
            ('fscore', 'F-Score', 'F-Score'),
            ('mcc', 'Matthews Correlation Coefficient (MCC)', 'MCC'),
        ]
        
        # Add precision/recall if we have space
        if show_attention_dynamics:
            metrics_config.append(('precision_recall', 'Precision & Recall', 'Score'))
        
        # Calculate subplot layout
        n_plots = len(metrics_config) + (1 if show_attention_dynamics and 'plot_batch_scores' in history else 0)
        n_cols = 3 if n_plots > 4 else 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Training History', fontsize=16)
        
        # Plot each metric generically
        for i, (metric_key, title, ylabel) in enumerate(metrics_config[:-1] if len(metrics_config) > 4 else metrics_config):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            if metric_key == 'precision_recall':
                # Special case for precision/recall combination
                self._plot_multiple_metrics(ax, history, 
                    [('train_precision', 'val_precision', 'Precision', 'b-', 'r-'),
                     ('train_recall', 'val_recall', 'Recall', 'b--', 'r--')],
                    title, ylabel)
            else:
                # Generic single metric plotting
                self._plot_metric(ax, history, f'train_{metric_key}', f'val_{metric_key}', title, ylabel)
        
        # Add prediction dynamics if available
        if show_attention_dynamics and 'plot_batch_scores' in history:
            dynamics_idx = len(metrics_config) - (1 if len(metrics_config) > 4 else 0)
            row, col = dynamics_idx // n_cols, dynamics_idx % n_cols
            self._plot_prediction_dynamics(axes[row, col], history['plot_batch_scores'], history['plot_batch_labels'])
        
        # Hide empty subplots
        for i in range(n_plots, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training history plot saved to: {plot_path}")
    
    def _plot_metric(
        self, 
        ax: plt.Axes, 
        history: Dict[str, List[float]], 
        train_key: str, 
        val_key: str, 
        title: str, 
        ylabel: str
    ) -> None:
        """Generic method to plot a single metric."""
        if train_key in history:
            ax.plot(history[train_key], 'b-', label='Training', linewidth=2)
        if val_key in history:
            ax.plot(history[val_key], 'r-', label='Validation', linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    
    def _plot_multiple_metrics(
        self, 
        ax: plt.Axes, 
        history: Dict[str, List[float]], 
        metrics_specs: List[Tuple[str, str, str, str, str]], 
        title: str, 
        ylabel: str
    ) -> None:
        """Generic method to plot multiple metrics on the same axis."""
        for train_key, val_key, label, train_style, val_style in metrics_specs:
            if train_key in history:
                ax.plot(history[train_key], train_style, label=f'Training {label}', linewidth=2)
            if val_key in history:
                ax.plot(history[val_key], val_style, label=f'Validation {label}', linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    
    def _plot_prediction_dynamics(
        self,
        ax: plt.Axes,
        plot_batch_scores: List[torch.Tensor],
        plot_batch_labels: torch.Tensor
    ) -> None:
        """Plot prediction dynamics of test mini batch over epochs."""
        pos_label, neg_label = False, False
        
        for i in range(len(plot_batch_labels)):
            # Convert scores to probabilities
            score_sequence = []
            for epoch_scores in plot_batch_scores:
                probs = F.softmax(epoch_scores, dim=1)
                if i < probs.shape[0]:
                    score_sequence.append(probs[i].cpu().numpy())
                else:
                    score_sequence.append([0.5, 0.5])  # Default if missing
            
            # Plot positive class (mutations)
            if plot_batch_labels[i]:
                pos_scores = [x[1] for x in score_sequence]  # Probability of positive class
                if not pos_label:
                    ax.plot(pos_scores, 'b-', label='Pos', linewidth=2)
                    pos_label = True
                else:
                    ax.plot(pos_scores, 'b-', linewidth=1)
            else:
                # Plot negative class (non-mutations)
                neg_scores = [x[0] for x in score_sequence]  # Probability of negative class
                if not neg_label:
                    ax.plot(neg_scores, 'r-', label='Neg', linewidth=2)
                    neg_label = True
                else:
                    ax.plot(neg_scores, 'r-', linewidth=1)
        
        ax.set_title('Prediction Dynamics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True)
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        save_name: str = 'attention_weights.png'
    ) -> None:
        """Plot attention weights in a grid."""
        logging.info("Creating attention weights visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # Convert to numpy if needed
        if isinstance(attention_weights, torch.Tensor):
            weights = attention_weights.cpu().numpy()
        else:
            weights = attention_weights
        
        # Create heatmap
        cax = plt.matshow(weights, cmap='bone')
        plt.colorbar(cax)
        
        # Formatting
        plt.grid(False)
        plt.xlabel('Time Periods', fontsize=12)
        plt.ylabel('Examples', fontsize=12)
        plt.title('Attention Weights', fontsize=14)
        
        # Save plot
        plot_path = self.results_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Attention weights plot saved to: {plot_path}")
    
    def plot_roc_curve(
        self,
        fpr: List[float],
        tpr: List[float],
        roc_auc: float,
        include_baseline: bool = True,
        baseline_fpr: Optional[List[float]] = None,
        baseline_tpr: Optional[List[float]] = None,
        baseline_auc: Optional[float] = None,
        save_name: str = 'roc_curve.png'
    ) -> None:
        """Plot ROC curve with optional baseline comparison."""
        logging.info("Creating ROC curve plot...")
        
        plt.figure(figsize=(8, 6))
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        # Plot main ROC curve
        plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'RNN Model (AUC = {roc_auc:.3f})')
        
        # Plot baseline if provided
        if include_baseline and baseline_fpr and baseline_tpr and baseline_auc:
            plt.plot(baseline_fpr, baseline_tpr, 'r-', linewidth=2, 
                    label=f'Logistic Regression (AUC = {baseline_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save plot
        plot_path = self.results_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"ROC curve plot saved to: {plot_path}")
    
    def plot_confusion_matrix(
        self,
        conf_matrix: List[List[int]],
        class_names: List[str] = None,
        save_name: str = 'confusion_matrix.png'
    ) -> None:
        """Plot confusion matrix."""
        logging.info("Creating confusion matrix plot...")
        
        if class_names is None:
            class_names = ['Non-Mutated', 'Mutated']
        
        plt.figure(figsize=(8, 6))
        
        conf_matrix_array = np.array(conf_matrix)
        im = plt.imshow(conf_matrix_array, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=14)
        plt.colorbar(im)
        
        # Add labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = conf_matrix_array.max() / 2.
        for i, j in np.ndindex(conf_matrix_array.shape):
            color = "white" if conf_matrix_array[i, j] > thresh else "black"
            plt.text(j, i, format(conf_matrix_array[i, j], 'd'),
                    horizontalalignment="center",
                    color=color, fontsize=12)
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Confusion matrix plot saved to: {plot_path}")
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_name: str = 'metrics_comparison.png'
    ) -> None:
        """Plot comparison of different models' metrics."""
        logging.info("Creating metrics comparison plot...")
        
        metric_names = ['accuracy', 'precision', 'recall', 'fscore', 'mcc']
        model_names = list(metrics_dict.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            values = [metrics_dict[model_name].get(metric, 0) for metric in metric_names]
            ax.bar(x + i * width, values, width, label=model_name)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.title() for m in metric_names])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / save_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Metrics comparison plot saved to: {plot_path}")
    
    def show_plots(self) -> None:
        """Display all plots (if running interactively)."""
        plt.show()
    
    def close_all_plots(self) -> None:
        """Close all matplotlib figures to free memory."""
        plt.close('all')