#!/usr/bin/env python3
"""
SARS-CoV-2 Spike Protein Mutation Prediction Pipeline

This is the main entry point for the amino acid mutation prediction pipeline.
The pipeline processes FASTA sequence data through clustering and creates
datasets for training RNN models.
"""

import argparse
import logging
import sys
from pathlib import Path

from scripts.config import load_config
from scripts.utils import setup_logger

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="SARS-CoV-2 Spike Protein Mutation Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py prepare --config configs/sars_cov_2_default.yaml
  python main.py cluster --config configs/sars_cov_2_default.yaml
  python main.py dataset --config configs/sars_cov_2_default.yaml
  python main.py train --config configs/sars_cov_2_default.yaml
  python main.py full-pipeline --config configs/sars_cov_2_default.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='configs/sars_cov_2_default.yaml',
        help='Path to configuration YAML file (default: configs/sars_cov_2_default.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Create subparsers for different pipeline commands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Pipeline commands')
    
    # Data preparation step
    subparsers.add_parser(
        'prepare',
        help='Step 1: Prepare and group raw FASTA data into time periods'
    )
    
    # Clustering step
    subparsers.add_parser(
        'cluster', 
        help='Step 2 & 3: Create clusters within periods and link them across time'
    )
    
    # Dataset creation step
    subparsers.add_parser(
        'dataset',
        help='Step 4: Create final training dataset from linked clusters'
    )
    
    # Training step
    subparsers.add_parser(
        'train',
        help='Step 5: Train and evaluate the RNN model'
    )
    
    # Full pipeline
    subparsers.add_parser(
        'full-pipeline',
        help='Run all pipeline steps sequentially'
    )
    
    return parser

def run_prepare_step(config: dict) -> None:
    """Run data preparation step."""
    try:
        # Import here to avoid circular imports and only when needed
        from scripts.pipeline.prepare_data import run
        logging.info("Starting data preparation step...")
        run(config)
        logging.info("Data preparation completed successfully.")
    except ImportError:
        logging.error("Data preparation module not found. Make sure scripts.pipeline.prepare_data exists.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Data preparation failed: {e}")
        sys.exit(1)

def run_cluster_step(config: dict) -> None:
    """Run clustering and cluster linking steps."""
    try:
        # Import here to avoid circular imports and only when needed
        from scripts.pipeline.create_clusters import run as run_clustering
        from scripts.pipeline.link_clusters import run as run_linking
        
        logging.info("Starting clustering step...")
        run_clustering(config)
        logging.info("Clustering completed successfully.")
        
        logging.info("Starting cluster linking step...")
        run_linking(config)
        logging.info("Cluster linking completed successfully.")
        
    except ImportError as e:
        logging.error(f"Clustering modules not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        sys.exit(1)

def run_dataset_step(config: dict) -> None:
    """Run dataset creation step."""
    try:
        # Import here to avoid circular imports and only when needed
        from scripts.pipeline.create_dataset import run
        logging.info("Starting dataset creation step...")
        run(config)
        logging.info("Dataset creation completed successfully.")
    except ImportError:
        logging.error("Dataset creation module not found. Make sure scripts.pipeline.create_dataset exists.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Dataset creation failed: {e}")
        sys.exit(1)

def run_train_step(config: dict) -> None:
    """Run model training and evaluation step."""
    try:
        # Import here to avoid circular imports and only when needed
        from scripts.pipeline.train_model import run
        logging.info("Starting model training step...")
        run(config)
        logging.info("Model training completed successfully.")
    except ImportError:
        logging.error("Training module not found. Make sure scripts.pipeline.train_model exists.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        sys.exit(1)

def run_full_pipeline(config: dict) -> None:
    """Run the complete pipeline from start to finish."""
    logging.info("Starting full pipeline execution...")
    
    # Run all steps in sequence
    run_prepare_step(config)
    run_cluster_step(config)
    run_dataset_step(config)
    run_train_step(config)
    
    logging.info("Full pipeline completed successfully!")

def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(verbose=args.verbose)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Execute the requested command
    if args.command == 'prepare':
        run_prepare_step(config)
    elif args.command == 'cluster':
        run_cluster_step(config)
    elif args.command == 'dataset':
        run_dataset_step(config)
    elif args.command == 'train':
        run_train_step(config)
    elif args.command == 'full-pipeline':
        run_full_pipeline(config)
    else:
        logging.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()