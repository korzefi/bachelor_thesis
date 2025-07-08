import argparse
import subprocess
import logging
import os

def run_pipeline(args):
    logging.basicConfig(level=logging.INFO)
    # 1. Train model
    logging.info('Step 1: Training model...')
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'train_model.py'),
        '--model_type', args.model_type,
        '--train_file', args.train_file,
        '--valid_file', args.valid_file,
        '--model_out', args.model_out,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--attn_seq_length', str(args.attn_seq_length) if args.attn_seq_length else 'None'
    ], check=True)
    # 2. Optionally, evaluate model (user can run evaluate_model.py separately)
    logging.info('Training pipeline completed successfully.')

def main():
    parser = argparse.ArgumentParser(description='Run the full training pipeline.')
    parser.add_argument('--model_type', choices=['rnn', 'attn'], required=True)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--valid_file', required=True)
    parser.add_argument('--model_out', required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--attn_seq_length', type=int, default=None)
    args = parser.parse_args()
    run_pipeline(args)

if __name__ == '__main__':
    main()
