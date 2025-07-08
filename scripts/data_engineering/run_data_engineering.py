import argparse
import logging
import os
import yaml 
from types import SimpleNamespace

def run_pipeline(args):
    logging.basicConfig(level=logging.INFO)

    # Import modules directly
    from scripts.data_engineering.split_fasta import split_fasta
    from scripts.data_engineering.fasta_to_csv import fasta_to_csv
    from scripts.data_engineering.clean_csv import clean_csv_dir
    from scripts.data_engineering.distribute_periods import distribute_periods

    # 1. Split FASTA
    split_dir = args.split_dir
    logging.info('Step 1: Splitting FASTA...')
    split_fasta(
        input_fasta=args.input_fasta,
        output_dir=split_dir,
        lines_per_file=args.lines_per_file,
        start_line=args.start_line,
        max_files=args.max_files
    )

    # 2. FASTA to CSV
    csv_dir = args.csv_dir
    logging.info('Step 2: Converting FASTA to CSV...')
    fasta_to_csv(
        fasta_dir=split_dir,
        csv_dir=csv_dir
    )

    # 3. Clean CSV
    cleaned_dir = args.cleaned_dir
    os.makedirs(cleaned_dir, exist_ok=True)
    for f in os.listdir(csv_dir):
        if f.endswith('.csv'):
            src = os.path.join(csv_dir, f)
            dst = os.path.join(cleaned_dir, f)
            if not os.path.exists(dst):
                import shutil
                shutil.copy(src, dst)
    logging.info('Step 3: Cleaning CSVs...')
    clean_csv_dir(
        csv_dir=cleaned_dir,
        min_len=args.min_len,
        max_len=args.max_len
    )

    # 4. Distribute periods
    periods_dir = args.periods_dir
    logging.info('Step 4: Distributing into periods...')
    distribute_periods(
        csv_dir=cleaned_dir,
        output_dir=periods_dir,
        division=args.division
    )
    logging.info('Data engineering pipeline completed successfully.')

def build_parser():
    parser = argparse.ArgumentParser(description='Run the full data engineering pipeline.')
    parser.add_argument('--config', type=str, help='Path to YAML config file (optional)')
    # All possible pipeline arguments
    pipeline_args = [
        ('input_fasta', None),
        ('split_dir', None),
        ('csv_dir', None),
        ('cleaned_dir', None),
        ('periods_dir', None),
        ('lines_per_file', 100000),
        ('start_line', 1),
        ('max_files', 50),
        ('min_len', 1260),
        ('max_len', 1280),
        ('division', 'month'),
    ]
    for arg, default in pipeline_args:
        arg_type = int if isinstance(default, int) else str
        parser.add_argument(f'--{arg}', type=arg_type, default=None)
    return parser, pipeline_args

def load_config(config_path):
    config = {}
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    return config

def get_param(args, config, key, default):
    val = getattr(args, key)
    if val is not None and val != 'None':
        return val
    return config.get(key, default)

def merge_args(args, config, pipeline_args):
    return SimpleNamespace(**{
        key: (int(get_param(args, config, key, default)) if isinstance(default, int) else get_param(args, config, key, default))
        for key, default in pipeline_args
    })


def main():
    parser, pipeline_args = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    merged_args = merge_args(args, config, pipeline_args)
    run_pipeline(merged_args)

if __name__ == '__main__':
    main()
