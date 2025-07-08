import argparse
import subprocess
import logging
import os
import yaml

def run_inference(args):
    logging.basicConfig(level=logging.INFO)
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'infer_model.py'),
        '--model_type', args.model_type,
        '--model_path', args.model_path,
        '--input_file', args.input_file,
        '--output_file', args.output_file,
        '--attn_seq_length', str(args.attn_seq_length) if args.attn_seq_length else 'None'
    ], check=True)
    logging.info('Inference completed successfully.')

def main():
    parser = argparse.ArgumentParser(description='Run the full inference pipeline.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (optional)')
    parser.add_argument('--model_type', choices=['rnn', 'attn'], default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--input_file', default=None)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--attn_seq_length', type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    def get_param(key, default=None):
        return getattr(args, key) if getattr(args, key) not in [None, 'None'] else config.get(key, default)

    class Args:
        pass
    merged_args = Args()
    merged_args.model_type = get_param('model_type', 'rnn')
    merged_args.model_path = get_param('model_path')
    merged_args.input_file = get_param('input_file')
    merged_args.output_file = get_param('output_file')
    merged_args.attn_seq_length = None if get_param('attn_seq_length') in [None, 'null', 'None'] else int(get_param('attn_seq_length'))

    run_inference(merged_args)

if __name__ == '__main__':
    main()
