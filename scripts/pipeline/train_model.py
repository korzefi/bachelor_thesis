#!/usr/bin/env python3
"""
Model Training Pipeline Orchestrator

This module orchestrates the complete training pipeline by coordinating:
- Dataset loading and preprocessing
- Model selection and initialization (user-configurable)
- Training with comprehensive metrics
- Flexible model saving with unique names and performance thresholds
- Evaluation and visualization
- Model saving and results export

Key Features:
- Fully config-driven (no hardcoded hyperparameters)
- Flexible model selection (user can specify any model type or provide custom models)
- All original evaluation metrics preserved (MCC, precision, recall, F-score, ROC)
- Smart model saving: best model or performance threshold-based
- Unique model names with architecture, date, and performance info
- Comprehensive visualization and result export
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch

from scripts.training.architectures import create_model
from scripts.training.dataset_processor import DatasetProcessor
from scripts.training.trainer import ModelTrainer
from scripts.training.evaluator import ModelEvaluator
from scripts.training.visualizer import TrainingVisualizer
import scripts.utils as utils


class ModelTrainingPipeline:
    """Main orchestrator for the model training pipeline."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize pipeline with configuration."""
        self.config = config
        self.data_config = config['data']
        self.train_config = config['train']
        self.model_config = config['models']
        self.results_config = config['results']
        
        # Validate required config sections
        self._validate_config()
        
        # Initialize components
        self.dataset_processor = DatasetProcessor(config)
        self.trainer = ModelTrainer(self.train_config['hyperparameters'][self.train_config['model_type']])
        self.evaluator = ModelEvaluator()
        self.visualizer = TrainingVisualizer(self.results_config['results_dir'])
        
        # Set up base paths (actual paths will be generated dynamically)
        self.base_model_dir = Path(self.model_config['model_dir'])
        self.base_results_dir = Path(self.results_config['results_dir'])
        
        # Model saving configuration
        self.save_config = self.train_config.get('model_saving', {})
        self.save_strategy = self.save_config.get('strategy', 'best')  # 'best', 'threshold', 'both'
        self.mcc_threshold = self.save_config.get('mcc_threshold', 0.5)
        self.always_save_best = self.save_config.get('always_save_best', True)
        
        # Current run timestamp for unique naming
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        required_fields = {
            'train.model_type': self.train_config.get('model_type'),
            'train.hyperparameters': self.train_config.get('hyperparameters'),
            'models.model_dir': self.model_config.get('model_dir'),
            'results.results_dir': self.results_config.get('results_dir')
        }
        
        missing_fields = [field for field, value in required_fields.items() if value is None]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        # Validate model type is supported or user provides custom model
        model_type = self.train_config['model_type']
        hyperparams = self.train_config['hyperparameters']
        
        if model_type not in hyperparams:
            raise ValueError(f"No hyperparameters found for model type '{model_type}'. "
                           f"Available: {list(hyperparams.keys())}")
        
        # Validate hyperparameters are present
        required_hyperparams = ['learning_rate', 'batch_size', 'epochs']
        model_hyperparams = hyperparams[model_type]
        missing_hyperparams = [hp for hp in required_hyperparams if hp not in model_hyperparams]
        
        if missing_hyperparams:
            raise ValueError(f"Missing required hyperparameters for {model_type}: {missing_hyperparams}")
    
    def run(self) -> None:
        """Execute the complete training pipeline."""
        logging.info("="*50)
        logging.info("STARTING MODEL TRAINING PIPELINE")
        logging.info(f"Run ID: {self.run_timestamp}")
        logging.info("="*50)
        
        try:
            # Step 1: Load and prepare dataset
            logging.info("Step 1: Loading and preparing dataset...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.dataset_processor.load_and_split_dataset()
            
            # Get dataset info for model initialization
            dataset_info = self.dataset_processor.get_dataset_info(X_train)
            logging.info(f"Dataset info: {dataset_info}")
            
            # Step 2: Initialize model
            logging.info("Step 2: Initializing model...")
            model = self._initialize_model(dataset_info)
            
            # Step 3: Optional logistic regression baseline
            if self.train_config.get('include_baseline', True):
                logging.info("Step 3: Computing logistic regression baseline...")
                baseline_results = self.evaluator.logistic_regression_baseline(
                    X_train, y_train, X_val, y_val,
                    window_size=self.train_config.get('window_size', 10)
                )
            else:
                baseline_results = None
            
            # Step 4: Train model with flexible saving
            logging.info("Step 4: Training model...")
            trained_model, training_history, saved_models = self._train_with_flexible_saving(
                model, X_train, y_train, X_val, y_val
            )
            
            # Step 5: Evaluate model
            logging.info("Step 5: Evaluating model...")
            if len(X_test) > 0:  # Only evaluate if test set exists
                test_results = self.evaluator.comprehensive_evaluation(trained_model, X_test, y_test)
            else:
                logging.warning("No test set available for evaluation")
                test_results = None
            
            # Step 6: Save final results
            logging.info("Step 6: Saving final results...")
            results_file = self._save_results(training_history, test_results, baseline_results, saved_models)
            
            # Step 7: Create visualizations
            logging.info("Step 7: Creating visualizations...")
            self._create_visualizations(training_history, test_results, baseline_results)
            
            logging.info("="*50)
            logging.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("="*50)
            
            # Print final summary
            self._print_final_summary(training_history, test_results, saved_models)
            
        except Exception as e:
            logging.error("="*50)
            logging.error("MODEL TRAINING PIPELINE FAILED!")
            logging.error(f"Error: {e}")
            logging.error("="*50)
            raise
    
    def _initialize_model(self, dataset_info: Dict[str, int]) -> torch.nn.Module:
        """Initialize model based on configuration."""
        model_type = self.train_config['model_type']
        hyperparameters = self.train_config['hyperparameters'][model_type]
        
        logging.info(f"Initializing {model_type} model...")
        logging.info(f"Model hyperparameters: {hyperparameters}")
        
        # Check if user wants to provide a custom model
        if model_type == 'custom':
            custom_model_path = self.train_config.get('custom_model_path')
            if custom_model_path:
                logging.info(f"Loading custom model from: {custom_model_path}")
                model = torch.load(custom_model_path)
                return model
            else:
                raise ValueError("Custom model type specified but no custom_model_path provided")
        
        # Use built-in model architectures
        model = create_model(
            model_type=model_type,
            seq_length=dataset_info['seq_length'],
            input_dim=dataset_info['input_dim'],
            output_dim=dataset_info['output_dim'],
            config=hyperparameters
        )
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"Model initialized: {param_count} total parameters, {trainable_params} trainable")
        
        return model
    
    def _train_with_flexible_saving(
        self,
        model: torch.nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> tuple[torch.nn.Module, Dict[str, Any], Dict[str, str]]:
        """Train model with flexible saving strategies."""
        logging.info(f"Model saving strategy: {self.save_strategy}")
        if self.save_strategy in ['threshold', 'both']:
            logging.info(f"MCC threshold for saving: {self.mcc_threshold}")
        
        # Train the model
        trained_model, training_history = self.trainer.train(
            model, X_train, y_train, X_val, y_val
        )
        
        # Keep track of saved models
        saved_models = {}
        
        # Get final metrics
        final_val_mcc = training_history['val_mcc'][-1] if training_history['val_mcc'] else 0.0
        best_val_mcc = max(training_history['val_mcc']) if training_history['val_mcc'] else 0.0
        
        # Save based on strategy
        if self.save_strategy == 'best' or (self.save_strategy == 'both' and self.always_save_best):
            # Save best model
            best_model_path = self._generate_model_path(
                model_type=self.train_config['model_type'],
                mcc_score=best_val_mcc,
                tag='best'
            )
            self._save_single_model(trained_model, best_model_path, best_val_mcc)
            saved_models['best'] = best_model_path
            logging.info(f"Best model saved (MCC: {best_val_mcc:.4f}): {best_model_path}")
        
        if self.save_strategy in ['threshold', 'both']:
            # Save if threshold is met
            if final_val_mcc >= self.mcc_threshold:
                threshold_model_path = self._generate_model_path(
                    model_type=self.train_config['model_type'],
                    mcc_score=final_val_mcc,
                    tag='threshold'
                )
                self._save_single_model(trained_model, threshold_model_path, final_val_mcc)
                saved_models['threshold'] = threshold_model_path
                logging.info(f"Threshold model saved (MCC: {final_val_mcc:.4f} >= {self.mcc_threshold}): {threshold_model_path}")
            else:
                logging.info(f"Threshold not met (MCC: {final_val_mcc:.4f} < {self.mcc_threshold}). Model not saved.")
        
        # Always save the current model for pipeline continuity (can be overwritten)
        current_model_path = self.model_config.get('model_path', str(self.base_model_dir / 'current_model.pth'))
        Path(current_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(trained_model, current_model_path)
        saved_models['current'] = current_model_path
        
        return trained_model, training_history, saved_models
    
    def _generate_model_path(
        self,
        model_type: str,
        mcc_score: float,
        tag: str = 'model'
    ) -> str:
        """Generate unique model path with architecture, timestamp, and performance info."""
        # Create filename components
        model_name = f"{model_type}_{tag}_mcc{mcc_score:.4f}_{self.run_timestamp}"
        filename = f"{model_name}.pth"
        
        # Full path
        model_path = self.base_model_dir / filename
        
        return str(model_path)
    
    def _save_single_model(
        self,
        model: torch.nn.Module,
        model_path: str,
        mcc_score: float
    ) -> None:
        """Save a single model with metadata."""
        # Ensure directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(model, model_path)
        
        # Save model state dict separately
        state_dict_path = model_path.replace('.pth', '_state_dict.pth')
        torch.save(model.state_dict(), state_dict_path)
        
        # Save metadata
        metadata = {
            'model_type': self.train_config['model_type'],
            'hyperparameters': self.train_config['hyperparameters'][self.train_config['model_type']],
            'validation_mcc': float(mcc_score),
            'timestamp': self.run_timestamp,
            'model_path': model_path,
            'state_dict_path': state_dict_path
        }
        
        metadata_path = model_path.replace('.pth', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_results(
        self, 
        training_history: Dict[str, Any], 
        test_results: Dict[str, Any] = None,
        baseline_results: Dict[str, Any] = None,
        saved_models: Dict[str, str] = None
    ) -> str:
        """Save comprehensive training and evaluation results."""
        # Generate unique results filename
        results_filename = f"training_results_{self.train_config['model_type']}_{self.run_timestamp}.json"
        results_file = self.base_results_dir / results_filename
        
        # Ensure directory exists
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Compile comprehensive results
        results = {
            'run_info': {
                'timestamp': self.run_timestamp,
                'model_type': self.train_config['model_type'],
                'run_id': self.run_timestamp
            },
            'config': {
                'model_type': self.train_config['model_type'],
                'hyperparameters': self.train_config['hyperparameters'][self.train_config['model_type']],
                'dataset_config': self.data_config,
                'training_config': self.train_config,
                'saving_config': self.save_config
            },
            'training_history': {
                key: value for key, value in training_history.items() 
                if key not in ['plot_batch_scores', 'plot_batch_labels']  # Exclude non-serializable data
            },
            'training_summary': self.trainer.get_training_summary(training_history),
            'saved_models': saved_models or {}
        }
        
        if test_results:
            results['test_results'] = {
                key: value for key, value in test_results.items()
                if key not in ['attention_weights']  # Exclude large arrays
            }
        
        if baseline_results:
            results['baseline_results'] = baseline_results
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str handles numpy arrays
        
        logging.info(f"Results saved to: {results_file}")
        return str(results_file)
    
    def _create_visualizations(
        self, 
        training_history: Dict[str, Any], 
        test_results: Dict[str, Any] = None,
        baseline_results: Dict[str, Any] = None
    ) -> None:
        """Create all visualizations with unique names."""
        timestamp_suffix = f"_{self.run_timestamp}"
        
        # Training history plots
        self.visualizer.plot_training_history(
            training_history, 
            show_attention_dynamics=True
        )
        
        # Rename to include timestamp
        old_path = self.visualizer.results_dir / 'training_history.png'
        new_path = self.visualizer.results_dir / f'training_history{timestamp_suffix}.png'
        if old_path.exists():
            old_path.rename(new_path)
        
        # Attention weights (if available)
        if (test_results and 'attention_weights' in test_results and 
            test_results['attention_weights'] is not None):
            attention_weights = torch.tensor(test_results['attention_weights'])
            self.visualizer.plot_attention_weights(
                attention_weights, 
                save_name=f'attention_weights{timestamp_suffix}.png'
            )
        
        # ROC curve
        if test_results and 'fpr' in test_results:
            baseline_fpr = baseline_results.get('fpr') if baseline_results else None
            baseline_tpr = baseline_results.get('tpr') if baseline_results else None
            baseline_auc = baseline_results.get('roc_auc') if baseline_results else None
            
            self.visualizer.plot_roc_curve(
                test_results['fpr'], 
                test_results['tpr'], 
                test_results['roc_auc'],
                include_baseline=baseline_results is not None,
                baseline_fpr=baseline_fpr,
                baseline_tpr=baseline_tpr,
                baseline_auc=baseline_auc,
                save_name=f'roc_curve{timestamp_suffix}.png'
            )
        
        # Confusion matrix
        if test_results and 'confusion_matrix' in test_results:
            self.visualizer.plot_confusion_matrix(
                test_results['confusion_matrix'],
                save_name=f'confusion_matrix{timestamp_suffix}.png'
            )
        
        # Metrics comparison (if baseline exists)
        if test_results and baseline_results:
            metrics_comparison = {
                'RNN Model': {
                    'accuracy': test_results['accuracy'],
                    'precision': test_results['precision'],
                    'recall': test_results['recall'],
                    'fscore': test_results['fscore'],
                    'mcc': test_results['mcc']
                },
                'Logistic Regression': baseline_results['val_metrics']
            }
            self.visualizer.plot_metrics_comparison(
                metrics_comparison,
                save_name=f'metrics_comparison{timestamp_suffix}.png'
            )
    
    def _print_final_summary(
        self, 
        training_history: Dict[str, Any], 
        test_results: Dict[str, Any] = None,
        saved_models: Dict[str, str] = None
    ) -> None:
        """Print final summary of results."""
        summary = self.trainer.get_training_summary(training_history)
        
        logging.info("\n" + "="*50)
        logging.info("FINAL RESULTS SUMMARY")
        logging.info("="*50)
        
        if summary:
            logging.info(f"Best Validation MCC: {summary.get('best_val_mcc', 'N/A'):.4f}")
            logging.info(f"Final Validation MCC: {summary.get('final_val_mcc', 'N/A'):.4f}")
            logging.info(f"Final Validation Accuracy: {summary.get('final_val_acc', 'N/A'):.4f}")
            logging.info(f"Final Validation F-Score: {summary.get('final_val_fscore', 'N/A'):.4f}")
        
        if test_results:
            logging.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
            logging.info(f"Test F-Score: {test_results['fscore']:.4f}")
            logging.info(f"Test MCC: {test_results['mcc']:.4f}")
            logging.info(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
        
        # Model saving summary
        if saved_models:
            logging.info("\nSaved Models:")
            for model_type, path in saved_models.items():
                if model_type != 'current':  # Don't show the temporary current model
                    logging.info(f"  {model_type.title()}: {path}")
        
        logging.info(f"\nRun ID: {self.run_timestamp}")
        logging.info(f"Results directory: {self.base_results_dir}")
        logging.info("="*50)


def run(config: Dict[str, Any]) -> None:
    """Main entry point for the model training pipeline."""
    pipeline = ModelTrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    # For testing purposes
    import sys
    sys.path.append('..')
    from scripts.config import load_config
    
    utils.setup_logger(verbose=True)
    config = load_config('../../configs/sars_cov_2_default.yaml')
    run(config)