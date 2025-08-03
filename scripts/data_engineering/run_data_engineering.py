import argparse
import logging
import os
import subprocess
import yaml

def run_pipeline(args):
    logging.basicConfig(level=logging.INFO)
    # 1. Split FASTA
    split_dir = args.split_dir
    logging.info('Step 1: Splitting FASTA...')
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'split_fasta.py'),
        '--input', args.input_fasta,
        '--output_dir', split_dir,
        '--lines_per_file', str(args.lines_per_file),
        '--start_line', str(args.start_line),
        '--max_files', str(args.max_files)
    ], check=True)
    # 2. FASTA to CSV
    csv_dir = args.csv_dir
    logging.info('Step 2: Converting FASTA to CSV...')
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'fasta_to_csv.py'),
        '--fasta_dir', split_dir,
        '--csv_dir', csv_dir
    ], check=True)
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
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'clean_csv.py'),
        '--csv_dir', cleaned_dir,
        '--expected_len', str(args.expected_len),
        '--error_margin', str(args.error_margin)
    ], check=True)
    # 4. Distribute periods
    periods_dir = args.periods_dir
    logging.info('Step 4: Distributing into periods...')
    subprocess.run([
        'python', os.path.join(os.path.dirname(__file__), 'distribute_periods.py'),
        '--csv_dir', cleaned_dir,
        '--output_dir', periods_dir,
        '--division', args.division
    ], check=True)
    logging.info('Data engineering pipeline completed successfully.')

def main():
    parser = argparse.ArgumentParser(description='Run the full data engineering pipeline.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (optional)')
    parser.add_argument('--input_fasta', default=None)
    parser.add_argument('--split_dir', default=None)
    parser.add_argument('--csv_dir', default=None)
    parser.add_argument('--cleaned_dir', default=None)
    parser.add_argument('--periods_dir', default=None)
    parser.add_argument('--lines_per_file', type=int, default=None)
    parser.add_argument('--start_line', type=int, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--expected_len', type=int, default=None)
    parser.add_argument('--error_margin', type=float, default=None)
    parser.add_argument('--division', default=None)
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
    merged_args.input_fasta = get_param('input_fasta')
    merged_args.split_dir = get_param('split_dir')
    merged_args.csv_dir = get_param('csv_dir')
    merged_args.cleaned_dir = get_param('cleaned_dir')
    merged_args.periods_dir = get_param('periods_dir')
    merged_args.lines_per_file = int(get_param('lines_per_file', 100000))
    merged_args.start_line = int(get_param('start_line', 1))
    merged_args.max_files = int(get_param('max_files', 50))
    merged_args.expected_len = int(get_param('expected_len', 1273))
    merged_args.error_margin = float(get_param('error_margin', 0.5))
    merged_args.division = get_param('division', 'month')

    run_pipeline(merged_args)

if __name__ == '__main__':
    main()
