#!/usr/bin/env python3
"""
Data Preparation Pipeline Module

This module handles the preparation of raw FASTA data by:
1. Splitting large FASTA files into manageable batches
2. Converting FASTA format to CSV
3. Cleaning sequences (removing ambiguous amino acids, wrong lengths, etc.)
4. Sorting sequences into time periods
5. Removing duplicates and empty files

Refactored from: scripts/preprocessing/grouping_raw_data.py
"""

import os
import subprocess
import shutil
import pandas as pd
import logging
from pathlib import Path
from natsort import natsorted
from typing import Dict, List

import scripts.utils as utils


class DataPreparationPipeline:
    """Main class that orchestrates the data preparation process."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration dictionary."""
        self.config = config
        self.data_config = config['data']
        self.prepare_config = config['prepare']
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directory structure."""
        dirs_to_create = [
            self.data_config['temp_fasta_dir'],
            self.data_config['temp_csv_dir'], 
            self.data_config['periods_dir'],
            self.data_config['periods_unique_dir']
        ]
        
        for dir_path in dirs_to_create:
            utils.create_dir(dir_path)
            logging.debug(f"Created directory: {dir_path}")
    
    def run(self) -> None:
        """Execute the complete data preparation pipeline."""
        logging.info("Starting data preparation pipeline...")
        
        try:
            # Step 1: Split large FASTA file into batches
            self._split_fasta_files()
            
            # Step 2: Convert FASTA to CSV format  
            self._convert_fasta_to_csv()
            
            # Step 3: Clean temporary FASTA files
            self._cleanup_temp_fasta()
            
            # Step 4: Clean CSV data (remove ambiguous, wrong length, etc.)
            self._clean_csv_data()
            
            # Step 5: Sort data into time periods
            self._sort_into_periods()
            
            # Step 6: Clean temporary CSV files
            self._cleanup_temp_csv()
            
            # Step 7: Remove duplicate entries within periods
            self._remove_period_duplicates()
            
            # Step 8: Remove empty period files
            self._remove_empty_periods()
            
            # Step 9: Create files with unique sequences
            self._create_unique_sequence_files()
            
            logging.info("Data preparation pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Data preparation pipeline failed: {e}")
            raise
    
    def _split_fasta_files(self) -> None:
        """Split large FASTA file into smaller batch files."""
        logging.info("Splitting FASTA files into batches...")
        
        split_config = self.prepare_config['split_fasta']
        lines_per_file = split_config['lines_per_file']
        max_files = split_config['max_files']
        
        raw_fasta_path = self.data_config['raw_fasta_file']
        temp_fasta_dir = self.data_config['temp_fasta_dir']
        
        # Get total number of lines in the raw file
        total_lines = self._get_file_line_count(raw_fasta_path)
        logging.info(f"Total lines in raw file: {total_lines}")
        
        # Calculate how many files we need, processing the entire file
        total_batches_needed = (total_lines + lines_per_file - 1) // lines_per_file  # Ceiling division
        actual_batches = min(max_files, total_batches_needed)
        
        logging.info(f"Processing entire file in {actual_batches} batches of up to {lines_per_file} lines each")
        
        # Split the file starting from line 1
        current_start = 1
        for i in range(actual_batches):
            current_end = min(current_start + lines_per_file - 1, total_lines)
            
            logging.info(f"Batch {i+1}/{actual_batches}: lines {current_start}-{current_end}")
            
            # Get configurable filename prefix
            fasta_prefix = self.prepare_config.get('fasta_prefix', 'batch_data')
            output_file = f"{temp_fasta_dir}/{fasta_prefix}-{current_start}-{current_end}.fasta"
            
            # Use sed to extract lines
            cmd = f"sed -n '{current_start},{current_end}p' {raw_fasta_path} > {output_file}"
            os.system(cmd)
            
            current_start = current_end + 1
            
            if current_start > total_lines:
                break
        
        logging.info(f"FASTA file splitting completed. Created {actual_batches} batch files.")
    
    def _convert_fasta_to_csv(self) -> None:
        """Convert FASTA batch files to CSV format."""
        logging.info("Converting FASTA files to CSV format...")
        
        temp_fasta_dir = self.data_config['temp_fasta_dir']
        temp_csv_dir = self.data_config['temp_csv_dir']
        
        # Get all FASTA files
        fasta_files = [f for f in os.listdir(temp_fasta_dir) if f.endswith('.fasta')]
        fasta_files = natsorted(fasta_files)
        
        for i, fasta_file in enumerate(fasta_files, 1):
            logging.info(f"Converting file {i}/{len(fasta_files)}: {fasta_file}")
            
            fasta_path = f"{temp_fasta_dir}/{fasta_file}"
            csv_file = fasta_file.replace('.fasta', '.csv')
            csv_path = f"{temp_csv_dir}/{csv_file}"
            
            # Read FASTA file
            df = pd.read_fwf(fasta_path, header=None)
            
            # Separate description and sequence lines
            description_df = df.iloc[::2, :].copy()
            description_df.columns = ['description']
            description_df.reset_index(drop=True, inplace=True)
            
            sequence_df = df.iloc[1::2, :].copy()
            sequence_df.columns = ['sequence']
            sequence_df.reset_index(drop=True, inplace=True)
            
            # Combine and save
            result_df = pd.concat([description_df, sequence_df], axis=1, join='inner')
            result_df.to_csv(csv_path, index=False)
        
        logging.info("FASTA to CSV conversion completed.")
    
    def _clean_csv_data(self) -> None:
        """Clean CSV data by removing invalid sequences."""
        logging.info("Cleaning CSV data...")
        
        temp_csv_dir = self.data_config['temp_csv_dir']
        clean_config = self.prepare_config['clean_sequences']
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(temp_csv_dir) if f.endswith('.csv')]
        csv_files = natsorted(csv_files)
        
        # Determine expected length and margins
        if 'expected_len' in clean_config and clean_config['expected_len'] is not None:
            # Use manually configured expected length
            expected_len = clean_config['expected_len']
            error_margin = clean_config.get('error_margin', 10)
            min_len = expected_len - error_margin
            max_len = expected_len + error_margin
            logging.info(f"Using configured expected length: {expected_len} ±{error_margin}")
        else:
            # Auto-detect expected length from data
            logging.info("Auto-detecting expected sequence length from data...")
            expected_len, min_len, max_len = self._auto_detect_sequence_length(csv_files, temp_csv_dir, clean_config)
            logging.info(f"Auto-detected expected length: {expected_len} (range: {min_len}-{max_len})")
        
        for i, csv_file in enumerate(csv_files, 1):
            logging.info(f"Cleaning file {i}/{len(csv_files)}: {csv_file}")
            
            csv_path = f"{temp_csv_dir}/{csv_file}"
            df = pd.read_csv(csv_path)
            
            # Apply cleaning steps
            df = self._remove_ambiguous_amino_acids(df)
            df = self._filter_by_length(df, min_len, max_len)
            df = self._parse_and_filter_description(df)
            df = self._remove_duplicates(df)
            
            # Save cleaned data
            df.to_csv(csv_path, index=False)
        
        logging.info("CSV data cleaning completed.")
    
    def _sort_into_periods(self) -> None:
        """Sort sequences into time periods."""
        logging.info("Sorting sequences into time periods...")
        
        temp_csv_dir = self.data_config['temp_csv_dir']
        division_technique = self.prepare_config['sort_periods']['division_technique']
        
        logging.info(f"Division technique: {division_technique}")
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(temp_csv_dir) if f.endswith('.csv')]
        csv_files = natsorted(csv_files)
        
        for i, csv_file in enumerate(csv_files, 1):
            logging.info(f"Processing file {i}/{len(csv_files)}: {csv_file}")
            
            csv_path = f"{temp_csv_dir}/{csv_file}"
            df = pd.read_csv(csv_path)
            
            # Sort by timestamp
            df = self._sort_by_timestamp(df)
            
            # Divide into periods
            self._divide_into_periods(df, division_technique)
        
        logging.info("Period sorting completed.")
    
    def _remove_period_duplicates(self) -> None:
        """Remove duplicate entries within each period."""
        logging.info("Removing duplicates within periods...")
        
        periods_dir = self.data_config['periods_dir']
        period_files = self._get_period_files()
        
        for period_file in period_files:
            df = pd.read_csv(period_file)
            original_count = len(df)
            
            df.drop_duplicates(subset=['isolate_name'], inplace=True)
            
            new_count = len(df)
            removed_count = original_count - new_count
            
            df.to_csv(period_file, index=False)
            
            file_name = os.path.basename(period_file)
            logging.info(f"Removed {removed_count} duplicates from {file_name}")
        
        logging.info("Period duplicate removal completed.")
    
    def _remove_empty_periods(self) -> None:
        """Remove empty period files."""
        logging.info("Removing empty period files...")
        
        period_files = self._get_period_files()
        removed_count = 0
        
        for period_file in period_files:
            df = pd.read_csv(period_file)
            
            if len(df) == 0:
                os.remove(period_file)
                file_name = os.path.basename(period_file)
                logging.info(f"Removed empty file: {file_name}")
                removed_count += 1
        
        logging.info(f"Removed {removed_count} empty period files.")
    
    def _create_unique_sequence_files(self) -> None:
        """Create files with unique sequences (removing sequence duplicates)."""
        logging.info("Creating files with unique sequences...")
        
        periods_dir = self.data_config['periods_dir']
        unique_dir = self.data_config['periods_unique_dir']
        
        # Get period files
        period_files = [f for f in os.listdir(periods_dir) if f.endswith('.csv')]
        period_files = natsorted(period_files)
        
        for period_file in period_files:
            input_path = f"{periods_dir}/{period_file}"
            output_path = f"{unique_dir}/{period_file}"
            
            df = pd.read_csv(input_path)
            original_count = len(df)
            
            # Remove sequence duplicates
            df.drop_duplicates(subset=['sequence'], inplace=True)
            
            new_count = len(df)
            removed_count = original_count - new_count
            
            df.to_csv(output_path, index=False)
            
            logging.info(f"Created {period_file} with unique sequences: {new_count} sequences (removed {removed_count} duplicates)")
        
        logging.info("Unique sequence files creation completed.")
    
    # Helper methods
    
    def _auto_detect_sequence_length(self, csv_files: List[str], temp_csv_dir: str, clean_config: Dict) -> tuple:
        """Auto-detect expected sequence length from data sample."""
        all_lengths = []
        sample_size = min(len(csv_files), 5)  # Sample first 5 files
        
        logging.info(f"Sampling {sample_size} files to determine sequence length distribution...")
        
        for csv_file in csv_files[:sample_size]:
            csv_path = f"{temp_csv_dir}/{csv_file}"
            df = pd.read_csv(csv_path)
            
            # Remove ambiguous sequences first for cleaner length detection
            df = self._remove_ambiguous_amino_acids(df)
            
            # Get sequence lengths
            lengths = df['sequence'].str.len().tolist()
            all_lengths.extend(lengths)
        
        if not all_lengths:
            raise ValueError("No valid sequences found for length detection")
        
        # Calculate statistics
        all_lengths = pd.Series(all_lengths)
        median_len = int(all_lengths.median())
        mode_len = int(all_lengths.mode()[0]) if not all_lengths.mode().empty else median_len
        std_len = all_lengths.std()
        
        logging.info(f"Length statistics - Median: {median_len}, Mode: {mode_len}, StdDev: {std_len:.2f}")
        
        # Use mode as expected length, fallback to median
        expected_len = mode_len
        
        # Calculate error margin: use configured margin or auto-calculate
        if 'error_margin' in clean_config:
            error_margin = clean_config['error_margin']
        else:
            # Use 2 standard deviations or minimum of 10
            error_margin = max(int(2 * std_len), 10)
        
        min_len = expected_len - error_margin
        max_len = expected_len + error_margin
        
        return expected_len, min_len, max_len
    
    def _get_file_line_count(self, file_path: str) -> int:
        """Get the number of lines in a file."""
        cmd = f'wc -l {file_path}'
        output = subprocess.check_output(cmd, shell=True, encoding='UTF-8')
        return int(output.split()[0])
    
    def _cleanup_temp_fasta(self) -> None:
        """Remove temporary FASTA directory."""
        logging.info("Cleaning up temporary FASTA files...")
        temp_fasta_dir = self.data_config['temp_fasta_dir']
        shutil.rmtree(temp_fasta_dir, ignore_errors=True)
    
    def _cleanup_temp_csv(self) -> None:
        """Remove temporary CSV directory."""
        logging.info("Cleaning up temporary CSV files...")
        temp_csv_dir = self.data_config['temp_csv_dir']
        shutil.rmtree(temp_csv_dir, ignore_errors=True)
    
    def _remove_ambiguous_amino_acids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove sequences with ambiguous amino acid codes."""
        ambiguous_aminos = ['B', 'J', 'Z', 'X', '-']
        df = df[~df['sequence'].str.contains('|'.join(ambiguous_aminos))]
        df.reset_index(drop=True, inplace=True)
        return df
    
    def _filter_by_length(self, df: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
        """Filter sequences by length."""
        # Must end with stop codon '*'
        df = df[df['sequence'].str.endswith('*')]
        
        # Filter by length
        df = df[(df['sequence'].str.len() >= min_len) & (df['sequence'].str.len() <= max_len)]
        
        return df
    
    def _parse_and_filter_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse description field and extract relevant information."""
        # Split description: gene|isolate_name|timestamp|rest
        splitted_desc = df['description'].str.split(pat='|', expand=True, n=3)
        splitted_desc.columns = ['gene', 'isolate_name', 'timestamp', 'rest']
        
        # Keep only isolate_name and timestamp
        splitted_desc = splitted_desc[['isolate_name', 'timestamp']].copy()
        
        # Adapt day format (add 1 to the last 2 digits)
        days = pd.to_numeric(splitted_desc['timestamp'].str[-2:]) + 1
        splitted_desc['timestamp'] = splitted_desc['timestamp'].str[:-2] + days.astype(str)
        
        # Replace description with parsed fields
        df = df.drop(columns=['description'])
        df = pd.concat([splitted_desc, df], axis=1, join='inner')
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries based on isolate_name."""
        df.drop_duplicates(subset=['isolate_name'], inplace=True)
        return df
    
    def _sort_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe by timestamp."""
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d', errors='coerce')
        df.sort_values(by='timestamp', inplace=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def _divide_into_periods(self, df: pd.DataFrame, division_technique: str) -> None:
        """Divide dataframe into time periods."""
        if division_technique == 'month':
            self._divide_by_month(df)
        elif division_technique == 'quarter':
            self._divide_by_quarter(df)
        elif division_technique == 'year':
            self._divide_by_year(df)
        else:
            raise ValueError(f"Unknown division technique: {division_technique}")
    
    def _divide_by_month(self, df: pd.DataFrame) -> None:
        """Divide sequences by month."""
        periods_dir = self.data_config['periods_dir']
        first_year = self._get_first_year(df)
        last_year = self._get_last_year(df)
        
        for year in range(first_year, last_year + 1):
            for month in range(1, 13):
                start_date = f'{year}-{month:02d}-01'
                
                # Get the last day of the month
                if month == 12:
                    end_date = f'{year}-12-31'
                else:
                    import calendar
                    last_day = calendar.monthrange(year, month)[1]
                    end_date = f'{year}-{month:02d}-{last_day}'
                
                period_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                
                if len(period_df) > 0:
                    output_path = f'{periods_dir}/{year}-{month}.csv'
                    header_flag = not os.path.exists(output_path)
                    period_df.to_csv(output_path, index=False, mode='a', header=header_flag)
    
    def _divide_by_quarter(self, df: pd.DataFrame) -> None:
        """Divide sequences by quarter."""
        periods_dir = self.data_config['periods_dir']
        first_year = self._get_first_year(df)
        last_year = self._get_last_year(df)
        
        quarters = [
            ['01-01', '03-31'], ['04-01', '06-30'], 
            ['07-01', '09-30'], ['10-01', '12-31']
        ]
        
        for year in range(first_year, last_year + 1):
            for i, (start, end) in enumerate(quarters, 1):
                start_date = f'{year}-{start}'
                end_date = f'{year}-{end}'
                
                period_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                
                if len(period_df) > 0:
                    output_path = f'{periods_dir}/{year}-q{i}.csv'
                    header_flag = not os.path.exists(output_path)
                    period_df.to_csv(output_path, index=False, mode='a', header=header_flag)
    
    def _divide_by_year(self, df: pd.DataFrame) -> None:
        """Divide sequences by year."""
        periods_dir = self.data_config['periods_dir']
        first_year = self._get_first_year(df)
        last_year = self._get_last_year(df)
        
        for year in range(first_year, last_year + 1):
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            
            period_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            if len(period_df) > 0:
                output_path = f'{periods_dir}/{year}.csv'
                header_flag = not os.path.exists(output_path)
                period_df.to_csv(output_path, index=False, mode='a', header=header_flag)
    
    def _get_first_year(self, df: pd.DataFrame) -> int:
        """Get the first year from the dataframe."""
        return df['timestamp'].dt.year.min()
    
    def _get_last_year(self, df: pd.DataFrame) -> int:
        """Get the last year from the dataframe."""
        return df['timestamp'].dt.year.max()
    
    def _get_period_files(self) -> List[str]:
        """Get list of period file paths."""
        periods_dir = self.data_config['periods_dir']
        files = [f for f in os.listdir(periods_dir) if f.endswith('.csv')]
        files = natsorted(files)
        return [f"{periods_dir}/{f}" for f in files]


def run(config: Dict) -> None:
    """Main entry point for the data preparation pipeline."""
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    # For testing purposes
    import sys
    sys.path.append('..')
    from scripts.config import load_config
    import scripts.utils as utils
    
    utils.setup_logger(verbose=True)
    config = load_config('../../configs/sars_cov_2_default.yaml')
    run(config)