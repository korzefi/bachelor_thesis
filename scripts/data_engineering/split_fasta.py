import os
import subprocess
import logging
import argparse
from natsort import natsorted


def split_fasta(input_fasta, output_dir, lines_per_file=100000, start_line=1, max_files=50):
    os.makedirs(output_dir, exist_ok=True)
    # Count lines in input file
    num_lines_command = f'wc -l {input_fasta}'
    output = subprocess.check_output(num_lines_command, shell=True, encoding='UTF-8')
    total_lines = int(output.split()[0])
    
    max_iters = min(max_files, ((total_lines - start_line + 1) // lines_per_file) + 1)
    current_start = start_line
    left_lines = total_lines - (start_line - 1)
    for i in range(max_iters):
        current_end = current_start + min(lines_per_file - 1, left_lines)
        out_file = os.path.join(output_dir, f'batch_{current_start}_{current_end}.fasta')
        copy_command = f'sed -n "{current_start},{current_end}p" {input_fasta} > {out_file}'
        logging.info(f'Splitting lines {current_start} to {current_end} into {out_file}')
        os.system(copy_command)
        current_start += lines_per_file
        left_lines = total_lines - (current_start - 1)
        if left_lines <= 0:
            break
    logging.info(f'Finished splitting {input_fasta} into batches in {output_dir}')


def main():
    parser = argparse.ArgumentParser(description="Split a large FASTA file into smaller batches.")
    parser.add_argument('--input', required=True, help='Path to input FASTA file')
    parser.add_argument('--output_dir', required=True, help='Directory to save split files')
    parser.add_argument('--lines_per_file', type=int, default=100000, help='Number of lines per split file')
    parser.add_argument('--start_line', type=int, default=1, help='Line to start splitting from')
    parser.add_argument('--max_files', type=int, default=50, help='Maximum number of split files')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    split_fasta(args.input, args.output_dir, args.lines_per_file, args.start_line, args.max_files)

if __name__ == '__main__':
    main()

