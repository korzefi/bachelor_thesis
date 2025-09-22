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
import numpy as np
import logging
import random
import json
from pathlib import Path
from natsort import natsorted
from typing import Dict, List
from datetime import datetime

import scripts.utils as utils
from scripts.utils import BatchProcessor, DataFrameChunker


class DataPreparationPipeline:
    """Main class that orchestrates the data preparation process."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration dictionary."""
        self.config = config
        self.data_config = config['data']
        self.prepare_config = config['prepare']
        self.error_config = config.get('error_handling', {})

        # Create directory structure
        self._create_directories()

        # Initialize memory monitoring
        memory_config = config.get('memory_optimization', {})
        self.batch_size = memory_config.get('data_preparation_batch_size', 1000)
        self.memory_monitor = utils.MemoryMonitor(
            memory_config.get('max_memory_mb'),
            memory_config.get('gc_frequency', 100)
        )

        # Initialize error tracking
        self.failed_files = []
        self.encoding_errors = []
        self.processing_errors = []
        self.successful_count = 0

        # Initialize progress tracking
        self.checkpoint_file = self.error_config.get('checkpoint_file', 'data/processed/pipeline_progress.json')
        self.enable_checkpoints = self.error_config.get('enable_checkpoints', False)
    
    def _create_directories(self) -> None:
        """Create necessary directory structure."""
        dirs_to_create = [
            self.data_config['temp_fasta_dir'],
            self.data_config['temp_csv_dir'],
            self.data_config['periods_dir'],
            self.data_config['periods_unique_dir']
        ]

        # Add quarantine directory if enabled
        if self.error_config.get('quarantine_corrupted_files', False):
            quarantine_dir = self.error_config.get('quarantine_dir', 'data/processed/quarantine')
            dirs_to_create.append(quarantine_dir)

        for dir_path in dirs_to_create:
            utils.create_dir(dir_path)
            logging.debug(f"Created directory: {dir_path}")
    
    def run(self) -> None:
        """Execute the complete data preparation pipeline with checkpoint support."""
        logging.info("Starting data preparation pipeline...")

        try:
            # Load checkpoint if available
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logging.info(f"Resuming from checkpoint: {checkpoint.get('completed_step', 'unknown')}")

            # Step 1: Split large FASTA file into batches
            if not self._is_step_completed('split_fasta'):
                logging.info("Step 1: Splitting FASTA files...")
                self._split_fasta_files()
                self._save_checkpoint('split_fasta')
            else:
                logging.info("Step 1: FASTA splitting already completed (skipped)")

            # Step 2: Convert FASTA to CSV format
            if not self._is_step_completed('convert_fasta_to_csv'):
                logging.info("Step 2: Converting FASTA to CSV...")
                self._convert_fasta_to_csv()
                self._save_checkpoint('convert_fasta_to_csv')
            else:
                logging.info("Step 2: FASTA to CSV conversion already completed (skipped)")

            # Step 3: Clean CSV data (remove ambiguous, wrong length, etc.)
            if not self._is_step_completed('clean_csv_data'):
                logging.info("Step 3: Cleaning CSV data...")
                self._clean_csv_data()
                self._save_checkpoint('clean_csv_data')
            else:
                logging.info("Step 3: CSV data cleaning already completed (skipped)")

            # Step 4: Sort data into time periods
            if not self._is_step_completed('sort_into_periods'):
                logging.info("Step 4: Sorting into time periods...")
                self._sort_into_periods()
                self._save_checkpoint('sort_into_periods')
            else:
                logging.info("Step 4: Period sorting already completed (skipped)")

            # Step 5: Remove duplicate entries within periods
            if not self._is_step_completed('remove_period_duplicates'):
                logging.info("Step 5: Removing period duplicates...")
                self._remove_period_duplicates()
                self._save_checkpoint('remove_period_duplicates')
            else:
                logging.info("Step 5: Period duplicate removal already completed (skipped)")

            # Step 6: Clean temporary FASTA files
            self._remove_temp_fasta()

            # Step 7: Clean temporary CSV files
            self._remove_temp_csv()

            # Step 8: Remove empty period files
            if not self._is_step_completed('remove_empty_periods'):
                logging.info("Step 8: Removing empty periods...")
                self._remove_empty_periods()
                self._save_checkpoint('remove_empty_periods')
            else:
                logging.info("Step 8: Empty period removal already completed (skipped)")

            # Step 9: Create files with unique sequences
            if not self._is_step_completed('create_unique_sequence_files'):
                logging.info("Step 9: Creating unique sequence files...")
                self._create_unique_sequence_files()
                self._save_checkpoint('create_unique_sequence_files')
            else:
                logging.info("Step 9: Unique sequence files already completed (skipped)")

            logging.info("Data preparation pipeline completed successfully!")

            # Clean up checkpoint file on successful completion
            if self.enable_checkpoints and os.path.exists(self.checkpoint_file):
                try:
                    os.remove(self.checkpoint_file)
                    logging.debug("Cleaned up checkpoint file after successful completion")
                except Exception as e:
                    logging.warning(f"Failed to clean up checkpoint file: {e}")

        except Exception as e:
            logging.error(f"Data preparation pipeline failed: {e}")
            # Save error checkpoint
            self._save_checkpoint('failed', metadata={'error': str(e), 'error_type': type(e).__name__})
            raise
    
    def _split_fasta_files(self) -> None:
        """Split large FASTA file into smaller batch files."""
        logging.info("Splitting FASTA files into batches...")
        
        split_config = self.prepare_config['split_fasta']
        start_line = split_config.get('start_line', 1)
        lines_per_file = split_config['lines_per_file']
        max_files = split_config['max_files']
        
        raw_fasta_path = self.data_config['raw_fasta_file']
        temp_fasta_dir = self.data_config['temp_fasta_dir']
        
        # Get total number of lines in the raw file
        total_lines = self._get_file_line_count(raw_fasta_path)
        logging.info(f"Total lines in raw file: {total_lines}")
        logging.info(f"Starting from line: {start_line}")
        
        # Calculate how many files we can create from the starting position
        remaining_lines = total_lines - start_line + 1
        total_batches_needed = (remaining_lines + lines_per_file - 1) // lines_per_file  # Ceiling division
        actual_batches = min(max_files, total_batches_needed)
        
        logging.info(f"Processing {remaining_lines} lines in {actual_batches} batches of up to {lines_per_file} lines each")
        
        # Split the file starting from the configured start_line
        current_start = start_line
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
        
        # Calculate and log the next start_line for subsequent runs
        next_start_line = start_line + (actual_batches * lines_per_file)
        if next_start_line <= total_lines:
            logging.info(f"For next run, use start_line: {next_start_line}")
        else:
            logging.info("File processing completed. No more lines to process.")
        
        logging.info(f"FASTA file splitting completed. Created {actual_batches} batch files.")
    
    def _convert_fasta_to_csv(self) -> None:
        """Convert FASTA batch files to CSV format with robust error handling."""
        logging.info("Converting FASTA files to CSV format...")
        temp_fasta_dir = self.data_config['temp_fasta_dir']
        temp_csv_dir = self.data_config['temp_csv_dir']

        fasta_files = natsorted([f for f in os.listdir(temp_fasta_dir) if f.endswith('.fasta')])

        # Reset error tracking for this operation
        self.failed_files = []
        self.encoding_errors = []
        self.processing_errors = []
        self.successful_count = 0
        consecutive_failures = 0

        # Error handling configuration
        skip_corrupted = self.error_config.get('skip_corrupted_files', True)
        max_consecutive = self.error_config.get('max_consecutive_failures', 10)
        encoding_chain = self.error_config.get('encoding_fallback_chain', ['utf-8', 'latin-1', 'cp1252'])

        for i, fasta_file in enumerate(fasta_files, 1):
            logging.info(f"Converting file {i}/{len(fasta_files)}: {fasta_file}")

            fasta_path = f"{temp_fasta_dir}/{fasta_file}"
            csv_file = fasta_file.replace('.fasta', '.csv')
            csv_path = f"{temp_csv_dir}/{csv_file}"

            try:
                # Try to process the file with encoding fallback
                success = self._process_single_fasta_file(fasta_path, csv_path, encoding_chain)

                if success:
                    self.successful_count += 1
                    consecutive_failures = 0
                    logging.debug(f"Successfully converted {fasta_file}")
                else:
                    # File processing failed with all encodings
                    self.failed_files.append(fasta_file)
                    consecutive_failures += 1

                    if skip_corrupted:
                        logging.warning(f"Skipping corrupted file {fasta_file} (tried all encodings)")
                        self._quarantine_file(fasta_path, "encoding_failure")
                    else:
                        raise RuntimeError(f"Failed to process {fasta_file} with any encoding")

            except Exception as e:
                # Handle other processing errors
                error_info = {
                    'file': fasta_file,
                    'error': str(e),
                    'type': type(e).__name__,
                    'timestamp': datetime.now().isoformat()
                }
                self.processing_errors.append(error_info)
                consecutive_failures += 1

                logging.error(f"Failed to process {fasta_file}: {e}")

                if skip_corrupted:
                    logging.warning(f"Skipping file {fasta_file} due to processing error")
                    try:
                        self._quarantine_file(fasta_path, "processing_error")
                    except Exception as quarantine_error:
                        logging.error(f"Failed to quarantine file {fasta_file}: {quarantine_error}")
                else:
                    raise

            # Check for too many consecutive failures
            if consecutive_failures >= max_consecutive:
                error_msg = f"Stopping conversion: {consecutive_failures} consecutive failures (max: {max_consecutive})"
                logging.error(error_msg)
                if not skip_corrupted:
                    raise RuntimeError(error_msg)
                else:
                    logging.warning(f"{error_msg}. Continuing with remaining files.")
                    consecutive_failures = 0  # Reset to try remaining files

        # Generate conversion report
        self._generate_conversion_report(fasta_files)
        logging.info(f"FASTA to CSV conversion completed. Success: {self.successful_count}/{len(fasta_files)}")

        # Save intermediate checkpoint with processing statistics
        if self.enable_checkpoints:
            metadata = {
                'total_files': len(fasta_files),
                'successful_files': self.successful_count,
                'failed_files': len(self.failed_files),
                'encoding_errors': len(self.encoding_errors),
                'processing_errors': len(self.processing_errors)
            }
            self._save_checkpoint('convert_fasta_to_csv', metadata=metadata)
    
    def _clean_csv_data(self) -> None:
        """Clean CSV data by removing invalid sequences using memory-optimized processing."""
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
        
        # Check if streaming should be used
        memory_config = self.config.get('memory_optimization', {})
        use_streaming = memory_config.get('use_streaming', True)
        
        for i, csv_file in enumerate(csv_files, 1):
            logging.info(f"Cleaning file {i}/{len(csv_files)}: {csv_file}")

            csv_path = f"{temp_csv_dir}/{csv_file}"

            # Check if file exists (may have been skipped during conversion)
            if not os.path.exists(csv_path):
                logging.warning(f"CSV file not found (likely skipped during conversion): {csv_file}")
                continue

            try:
                # Check file size to determine processing method
                file_size_mb = os.path.getsize(csv_path) / 1024 / 1024

                if use_streaming and file_size_mb > 10:  # Use streaming for files > 10MB
                    self._clean_csv_file_streaming(csv_path, min_len, max_len)
                else:
                    self._clean_csv_file_standard(csv_path, min_len, max_len)

                # Memory monitoring
                self.memory_monitor.batch_completed()

            except Exception as e:
                if self.error_config.get('skip_corrupted_files', True):
                    logging.warning(f"Skipping cleaning of corrupted CSV file {csv_file}: {e}")
                    continue
                else:
                    logging.error(f"Failed to clean CSV file {csv_file}: {e}")
                    raise
        
        logging.info("CSV data cleaning completed.")
    
    def _clean_csv_file_streaming(self, csv_path: str, min_len: int, max_len: int) -> None:
        """Clean a single CSV file using streaming/chunked processing."""
        logging.info(f"Using streaming processing for large file: {os.path.basename(csv_path)}")
        
        temp_output = f"{csv_path}.temp"
        total_rows = 0
        cleaned_rows = 0
        
        try:
            # Process in chunks
            with open(temp_output, 'w') as outfile:
                header_written = False
                
                for chunk in DataFrameChunker.read_csv_in_chunks(csv_path, self.batch_size):
                    total_rows += len(chunk)

                    # Apply cleaning steps
                    chunk = self._remove_ambiguous_amino_acids(chunk)
                    chunk = self._filter_by_length(chunk, min_len, max_len)

                    # Only parse description if it hasn't been parsed already
                    if not self._has_parsed_description(chunk):
                        chunk = self._parse_and_filter_description(chunk)
                        logging.debug(f"Parsed description column for chunk in {os.path.basename(csv_path)}")
                    else:
                        logging.debug(f"Description already parsed for chunk in {os.path.basename(csv_path)}, skipping")

                    chunk = self._remove_duplicates(chunk)
                    
                    # Write cleaned chunk
                    if not chunk.empty:
                        DataFrameChunker.write_csv_incrementally(
                            chunk, temp_output, 
                            header=not header_written, mode='a'
                        )
                        header_written = True
                        cleaned_rows += len(chunk)
            
            # Replace original file with cleaned version
            shutil.move(temp_output, csv_path)
            
            logging.info(f"Streaming clean completed: {cleaned_rows}/{total_rows} sequences remaining")
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_output):
                os.remove(temp_output)
            logging.error(f"Streaming processing failed for {csv_path}: {e}")
            # Fallback to standard processing
            self._clean_csv_file_standard(csv_path, min_len, max_len)
    
    def _clean_csv_file_standard(self, csv_path: str, min_len: int, max_len: int) -> None:
        """Clean a single CSV file using memory-efficient chunk processing."""
        import gc

        temp_output = f"{csv_path}.temp"
        chunk_size = self.batch_size
        total_sequences = 0
        cleaned_sequences = 0

        with open(temp_output, 'w') as outfile:
            header_written = False
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                total_sequences += len(chunk)

                # Apply cleaning steps
                chunk = self._remove_ambiguous_amino_acids(chunk)
                chunk = self._filter_by_length(chunk, min_len, max_len)

                # Only parse description if it hasn't been parsed already
                if not self._has_parsed_description(chunk):
                    chunk = self._parse_and_filter_description(chunk)
                    logging.debug(f"Parsed description column for chunk in {os.path.basename(csv_path)}")
                else:
                    logging.debug(f"Description already parsed for chunk in {os.path.basename(csv_path)}, skipping")

                chunk = self._remove_duplicates(chunk)

                if not chunk.empty:
                    chunk.to_csv(outfile, mode='a', header=not header_written, index=False)
                    header_written = True
                    cleaned_sequences += len(chunk)
                    del chunk
                    gc.collect()

        shutil.move(temp_output, csv_path)
        logging.info(f"Standard clean completed: {cleaned_sequences}/{total_sequences} sequences remaining")

    def _sort_into_periods_parallel(self) -> None:
        """Sort sequences into time periods using multiprocessing with fault tolerance."""
        logging.info("Sorting sequences into time periods (parallel)...")

        from multiprocessing import Pool, cpu_count
        import functools

        temp_csv_dir = self.data_config['temp_csv_dir']
        division_technique = self.prepare_config['sort_periods']['division_technique']

        csv_files = natsorted([f for f in os.listdir(temp_csv_dir) if f.endswith('.csv')])

        # Prepare safe worker function with fixed parameters
        worker_func = functools.partial(
            self._safe_process_single_csv_file,
            temp_csv_dir=temp_csv_dir,
            division_technique=division_technique,
            periods_dir=self.data_config['periods_dir']
        )

        # Process files in parallel with error handling
        n_workers = min(cpu_count(), len(csv_files))
        logging.info(f"Using {n_workers} workers for {len(csv_files)} files")

        successful_files = []
        failed_files = []
        skipped_files = []

        with Pool(processes=n_workers) as pool:
            results = pool.map(worker_func, csv_files)

        # Process results
        for csv_file, success, message in results:
            if success:
                successful_files.append(csv_file)
                logging.debug(f"✅ {csv_file}: {message}")
            else:
                if "uncleaned structure" in message:
                    skipped_files.append((csv_file, message))
                    logging.warning(f"⏭️  {csv_file}: {message}")
                else:
                    failed_files.append((csv_file, message))
                    logging.error(f"❌ {csv_file}: {message}")

        # Report summary
        total_files = len(csv_files)
        success_count = len(successful_files)
        skip_count = len(skipped_files)
        fail_count = len(failed_files)

        logging.info(f"Period sorting summary: {success_count}/{total_files} successful, "
                    f"{skip_count} skipped (uncleaned), {fail_count} failed")

        if skip_count > 0:
            logging.warning(f"Skipped {skip_count} files with uncleaned structure. "
                           f"These files may need to be re-cleaned before period sorting.")

        if fail_count > 0:
            logging.error(f"Failed to process {fail_count} files:")
            for file, error in failed_files[:5]:  # Show first 5 errors
                logging.error(f"  {file}: {error}")
            if fail_count > 5:
                logging.error(f"  ... and {fail_count - 5} more")

        # Check if we have enough successful files to continue
        success_rate = success_count / total_files if total_files > 0 else 0
        if success_rate < 0.5:  # Less than 50% success rate
            raise RuntimeError(f"Period sorting failed for too many files: {success_rate:.1%} success rate. "
                             f"Only {success_count}/{total_files} files processed successfully.")

        logging.info(f"Period sorting completed with {success_rate:.1%} success rate")

    @staticmethod
    def _validate_and_fix_timestamps(timestamps: pd.Series) -> pd.Series:
        """Validate and fix malformed timestamps with comprehensive error handling."""
        import re
        import calendar

        def fix_timestamp(timestamp_str):
            """Fix timestamp format to ensure valid YYYY-MM-DD format."""
            if pd.isna(timestamp_str):
                return timestamp_str

            timestamp_str = str(timestamp_str).strip()

            # Remove any trailing characters after the date (handles "2023-06-052" -> "2023-06-05")
            timestamp_str = re.sub(r'^(\d{4}-\d{1,2}-\d{1,2}).*$', r'\1', timestamp_str)

            # Pattern to match YYYY-M-D or YYYY-MM-D formats
            pattern = r'^(\d{4})-(\d{1,2})-(\d{1,2})$'
            match = re.match(pattern, timestamp_str)

            if not match:
                return timestamp_str  # Return original if format doesn't match

            year_str, month_str, day_str = match.groups()

            try:
                year = int(year_str)
                month = int(month_str)
                day = int(day_str)

                # Fix invalid months
                if month == 0:
                    month = 1  # Convert month 00 to 01
                elif month > 12:
                    month = 12  # Convert month 13+ to 12

                # Fix invalid days for the given month/year
                if day == 0:
                    day = 1
                else:
                    # Get the maximum valid day for this month/year
                    max_day = calendar.monthrange(year, month)[1]
                    if day > max_day:
                        day = max_day  # Clamp to last valid day of month

                # Return properly formatted date
                return f"{year:04d}-{month:02d}-{day:02d}"

            except (ValueError, calendar.IllegalMonthError) as e:
                # If any conversion fails, return original string
                logging.debug(f"Could not fix timestamp '{timestamp_str}': {e}")
                return timestamp_str

        # Apply timestamp fixing
        fixed_timestamps = timestamps.apply(fix_timestamp)
        return fixed_timestamps

    @staticmethod
    def _safe_process_single_csv_file(csv_file: str, temp_csv_dir: str,
                                    division_technique: str, periods_dir: str) -> tuple:
        """Safe wrapper for _process_single_csv_file that handles errors gracefully."""
        try:
            csv_path = f"{temp_csv_dir}/{csv_file}"

            # Validate CSV structure before processing
            if not os.path.exists(csv_path):
                return (csv_file, False, f"File not found: {csv_path}")

            # Check file structure
            try:
                sample_df = pd.read_csv(csv_path, nrows=1)
                required_columns = ['isolate_name', 'timestamp', 'sequence']

                if not all(col in sample_df.columns for col in required_columns):
                    if 'description' in sample_df.columns:
                        return (csv_file, False, "File has uncleaned structure (description column), skipping period sorting")
                    else:
                        return (csv_file, False, f"Unexpected CSV structure: {list(sample_df.columns)}")

            except Exception as e:
                return (csv_file, False, f"Error reading CSV structure: {e}")

            # Process the file
            DataPreparationPipeline._process_single_csv_file(csv_file, temp_csv_dir, division_technique, periods_dir)
            return (csv_file, True, "Successfully processed")

        except Exception as e:
            return (csv_file, False, f"Processing failed: {e}")

    @staticmethod
    def _process_single_csv_file(csv_file: str, temp_csv_dir: str,
                                division_technique: str, periods_dir: str) -> None:
        """Process single CSV file for period sorting with enhanced error handling."""
        csv_path = f"{temp_csv_dir}/{csv_file}"

        total_rows_processed = 0
        total_rows_invalid = 0
        correction_stats = {'invalid_months': 0, 'invalid_days': 0, 'trailing_chars': 0}

        try:
            # Read in chunks to manage memory
            for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=10000)):
                original_count = len(chunk)
                total_rows_processed += original_count

                # Store original timestamps for comparison
                original_timestamps = chunk['timestamp'].copy()

                # Validate and fix timestamp format before parsing
                chunk['timestamp'] = DataPreparationPipeline._validate_and_fix_timestamps(chunk['timestamp'])

                # Count how many timestamps were corrected
                corrections = (original_timestamps != chunk['timestamp']).sum()
                if corrections > 0:
                    logging.debug(f"Corrected {corrections} malformed timestamps in {csv_file} chunk {chunk_idx}")

                # Sort by timestamp with multiple fallback strategies
                try:
                    # Primary strategy: strict format parsing
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='%Y-%m-%d')
                except ValueError as e:
                    # Secondary strategy: flexible format detection
                    logging.debug(f"Format parsing failed for {csv_file}, using flexible parsing: {str(e)[:100]}...")
                    try:
                        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='%Y-%m-%d', errors='coerce')
                    except Exception as e2:
                        # Tertiary strategy: full automatic detection
                        logging.debug(f"Flexible parsing failed for {csv_file}, using automatic detection: {str(e2)[:100]}...")
                        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', infer_datetime_format=True)

                # Handle and count invalid timestamps
                invalid_mask = chunk['timestamp'].isna()
                invalid_count = invalid_mask.sum()
                total_rows_invalid += invalid_count

                if invalid_count > 0:
                    logging.debug(f"Found {invalid_count}/{original_count} invalid timestamps in {csv_file} chunk {chunk_idx}")
                    chunk = chunk.dropna(subset=['timestamp']).copy()
                else:
                    # Ensure we're working with a copy even when no rows are dropped
                    chunk = chunk.copy()

                # Skip empty chunks
                if len(chunk) == 0:
                    logging.debug(f"Skipping empty chunk {chunk_idx} in {csv_file}")
                    continue

                chunk = chunk.sort_values(by='timestamp')

                # Divide into periods using .loc to be explicit about assignment
                if division_technique == 'month':
                    chunk.loc[:, 'period'] = chunk['timestamp'].dt.to_period('M')
                elif division_technique == 'quarter':
                    chunk.loc[:, 'period'] = chunk['timestamp'].dt.to_period('Q')
                elif division_technique == 'year':
                    chunk.loc[:, 'period'] = chunk['timestamp'].dt.to_period('Y')

                # Save to appropriate period files
                for period, group in chunk.groupby('period'):
                    output_file = f"{periods_dir}/{period}.csv"
                    mode = 'a' if os.path.exists(output_file) else 'w'
                    header = not os.path.exists(output_file)
                    group.to_csv(output_file, mode=mode, header=header, index=False)

            # Log processing summary
            if total_rows_invalid > 0:
                invalid_percentage = (total_rows_invalid / total_rows_processed) * 100
                logging.info(f"Processed {csv_file}: {total_rows_processed} total rows, "
                           f"{total_rows_invalid} invalid timestamps ({invalid_percentage:.1f}%)")
            else:
                logging.debug(f"Successfully processed {csv_file}: {total_rows_processed} rows, no timestamp issues")

        except Exception as e:
            logging.error(f"Failed to process {csv_file}: {e}")
            # Don't re-raise - let the safe wrapper handle the error reporting

    def _sort_into_periods(self) -> None:
        """Sort sequences into time periods."""
        # Check if parallel processing should be used
        performance_config = self.config.get('optimization', {}).get('performance', {})
        use_multiprocessing = performance_config.get('use_multiprocessing', False)

        if use_multiprocessing:
            self._sort_into_periods_parallel()
        else:
            self._sort_into_periods_sequential()

    def _sort_into_periods_sequential(self) -> None:
        """Sort sequences into time periods sequentially."""
        logging.info("Sorting sequences into time periods (sequential)...")

        temp_csv_dir = self.data_config['temp_csv_dir']
        division_technique = self.prepare_config['sort_periods']['division_technique']

        logging.info(f"Division technique: {division_technique}")

        # Get all CSV files
        csv_files = [f for f in os.listdir(temp_csv_dir) if f.endswith('.csv')]
        csv_files = natsorted(csv_files)

        for i, csv_file in enumerate(csv_files, 1):
            logging.info(f"Processing file {i}/{len(csv_files)}: {csv_file}")

            csv_path = f"{temp_csv_dir}/{csv_file}"

            # Check if file exists
            if not os.path.exists(csv_path):
                logging.warning(f"CSV file not found for period sorting: {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_path)

                if df.empty:
                    logging.warning(f"Empty CSV file: {csv_file}")
                    continue

                # Sort by timestamp
                df = self._sort_by_timestamp(df)

                # Divide into periods
                self._divide_into_periods(df, division_technique)

            except Exception as e:
                if self.error_config.get('skip_corrupted_files', True):
                    logging.warning(f"Skipping period sorting for corrupted file {csv_file}: {e}")
                    continue
                else:
                    logging.error(f"Failed to sort periods for {csv_file}: {e}")
                    raise

        logging.info("Period sorting completed.")
    
    def _remove_period_duplicates(self) -> None:
        """Remove duplicate entries within each period."""
        logging.info("Removing duplicates within periods...")
        
        periods_dir = self.data_config['periods_dir']
        period_files = self._get_period_files()
        
        for period_file in period_files:
            file_name = os.path.basename(period_file)

            try:
                if not os.path.exists(period_file):
                    logging.warning(f"Period file not found: {file_name}")
                    continue

                df = pd.read_csv(period_file)
                original_count = len(df)

                if original_count == 0:
                    logging.warning(f"Empty period file: {file_name}")
                    continue

                df.drop_duplicates(subset=['isolate_name'], inplace=True)

                new_count = len(df)
                removed_count = original_count - new_count

                df.to_csv(period_file, index=False)

                logging.info(f"Removed {removed_count} duplicates from {file_name}")

            except Exception as e:
                if self.error_config.get('skip_corrupted_files', True):
                    logging.warning(f"Skipping duplicate removal for corrupted file {file_name}: {e}")
                    continue
                else:
                    logging.error(f"Failed to remove duplicates from {file_name}: {e}")
                    raise
        
        logging.info("Period duplicate removal completed.")
    
    def _remove_empty_periods(self) -> None:
        """Remove empty period files."""
        logging.info("Removing empty period files...")
        
        period_files = self._get_period_files()
        removed_count = 0
        
        for period_file in period_files:
            file_name = os.path.basename(period_file)

            try:
                if not os.path.exists(period_file):
                    logging.debug(f"Period file already removed: {file_name}")
                    continue

                df = pd.read_csv(period_file)

                if len(df) == 0:
                    os.remove(period_file)
                    logging.info(f"Removed empty file: {file_name}")
                    removed_count += 1

            except Exception as e:
                logging.warning(f"Error checking/removing empty file {file_name}: {e}")
                continue
        
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

            try:
                if not os.path.exists(input_path):
                    logging.warning(f"Period file not found for unique sequences: {period_file}")
                    continue

                df = pd.read_csv(input_path)
                original_count = len(df)

                if original_count == 0:
                    logging.warning(f"Empty period file for unique sequences: {period_file}")
                    # Create empty output file to maintain consistency
                    df.to_csv(output_path, index=False)
                    continue

                # Remove sequence duplicates
                df.drop_duplicates(subset=['sequence'], inplace=True)

                new_count = len(df)
                removed_count = original_count - new_count

                df.to_csv(output_path, index=False)

                logging.info(f"Created {period_file} with unique sequences: {new_count} sequences (removed {removed_count} duplicates)")

            except Exception as e:
                if self.error_config.get('skip_corrupted_files', True):
                    logging.warning(f"Skipping unique sequences creation for corrupted file {period_file}: {e}")
                    continue
                else:
                    logging.error(f"Failed to create unique sequences for {period_file}: {e}")
                    raise
        
        logging.info("Unique sequence files creation completed.")
    
    # Helper methods
    
    def _auto_detect_sequence_length(self, csv_files: List[str], temp_csv_dir: str, clean_config: Dict) -> tuple:
        """Auto-detect expected sequence length from data sample."""
        all_lengths = []
        sample_size = min(len(csv_files), 10)  # Sample 10 files randomly
        
        # Randomly select files instead of taking the first ones
        sampled_files = random.sample(csv_files, sample_size)
        
        logging.info(f"Sampling {sample_size} randomly selected files to determine sequence length distribution...")
        
        max_sequences = 1000000
        sequences_collected = 0
        
        for csv_file in sampled_files:
            if sequences_collected >= max_sequences:
                break
                
            csv_path = f"{temp_csv_dir}/{csv_file}"
            df = pd.read_csv(csv_path)
            
            # Remove ambiguous sequences first for cleaner length detection
            df = self._remove_ambiguous_amino_acids(df)
            
            # Randomly sample sequences from this file if needed
            remaining_slots = max_sequences - sequences_collected
            if len(df) > remaining_slots:
                df = df.sample(n=remaining_slots, random_state=42)
            
            # Get sequence lengths
            lengths = df['sequence'].str.len().tolist()
            all_lengths.extend(lengths)
            sequences_collected += len(lengths)
        
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

    def _process_single_fasta_file(self, fasta_path: str, csv_path: str, encoding_chain: List[str]) -> bool:
        """Process single FASTA file with encoding fallback chain."""
        for encoding in encoding_chain:
            try:
                logging.debug(f"Trying encoding {encoding} for {os.path.basename(fasta_path)}")

                with open(csv_path, 'w', encoding='utf-8') as csvfile:
                    csvfile.write('description,sequence\n')

                    with open(fasta_path, 'r', encoding=encoding) as fastafile:
                        description = None
                        sequence = []

                        for line_num, line in enumerate(fastafile, 1):
                            try:
                                line = line.strip()
                                if line.startswith('>'):
                                    if description:
                                        csvfile.write(f'"{description}","{"".join(sequence)}"\n')
                                    description = line
                                    sequence = []
                                else:
                                    sequence.append(line)
                            except Exception as line_error:
                                logging.warning(f"Skipping corrupted line {line_num} in {os.path.basename(fasta_path)}: {line_error}")
                                continue

                        if description:
                            csvfile.write(f'"{description}","{"".join(sequence)}"\n')

                # File processed successfully
                logging.debug(f"Successfully processed {os.path.basename(fasta_path)} with encoding {encoding}")
                return True

            except UnicodeDecodeError as e:
                # Track encoding error
                error_info = {
                    'file': os.path.basename(fasta_path),
                    'encoding': encoding,
                    'error': str(e),
                    'position': e.start if hasattr(e, 'start') else 'unknown',
                    'timestamp': datetime.now().isoformat()
                }
                self.encoding_errors.append(error_info)
                logging.debug(f"Encoding {encoding} failed for {os.path.basename(fasta_path)}: {e}")
                continue

            except Exception as e:
                logging.debug(f"Processing failed with encoding {encoding} for {os.path.basename(fasta_path)}: {e}")
                continue

        # All encodings failed
        logging.error(f"All encodings failed for {os.path.basename(fasta_path)}")
        return False

    def _quarantine_file(self, file_path: str, reason: str) -> None:
        """Move corrupted file to quarantine directory."""
        if not self.error_config.get('quarantine_corrupted_files', False):
            return

        quarantine_dir = self.error_config.get('quarantine_dir', 'data/processed/quarantine')
        utils.create_dir(quarantine_dir)

        # Create subdirectory for the reason
        reason_dir = os.path.join(quarantine_dir, reason)
        utils.create_dir(reason_dir)

        # Move file to quarantine
        filename = os.path.basename(file_path)
        quarantine_path = os.path.join(reason_dir, filename)

        try:
            shutil.move(file_path, quarantine_path)
            logging.info(f"Quarantined {filename} to {quarantine_path} (reason: {reason})")
        except Exception as e:
            logging.error(f"Failed to quarantine {filename}: {e}")

    def _generate_conversion_report(self, total_files: List[str]) -> None:
        """Generate detailed error report for FASTA conversion."""
        if not self.error_config.get('save_error_report', False):
            return

        report = {
            'conversion_summary': {
                'total_files': len(total_files),
                'successful_files': self.successful_count,
                'failed_files': len(self.failed_files),
                'success_rate': self.successful_count / len(total_files) * 100 if total_files else 0,
                'timestamp': datetime.now().isoformat()
            },
            'failed_files': self.failed_files,
            'encoding_errors': self.encoding_errors,
            'processing_errors': self.processing_errors,
            'recommendations': self._generate_recommendations()
        }

        # Save report
        report_path = self.error_config.get('error_report_path', 'data/processed/error_report.json')
        utils.create_dir(os.path.dirname(report_path))

        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logging.info(f"Error report saved to {report_path}")
        except Exception as e:
            logging.error(f"Failed to save error report: {e}")

        # Log summary
        logging.info(f"Conversion Summary:")
        logging.info(f"  Total files: {len(total_files)}")
        logging.info(f"  Successful: {self.successful_count}")
        logging.info(f"  Failed: {len(self.failed_files)}")
        logging.info(f"  Success rate: {report['conversion_summary']['success_rate']:.1f}%")

        if self.encoding_errors:
            logging.warning(f"  Encoding errors: {len(self.encoding_errors)}")

        if self.processing_errors:
            logging.warning(f"  Processing errors: {len(self.processing_errors)}")

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []

        if self.encoding_errors:
            encoding_counts = {}
            for error in self.encoding_errors:
                enc = error.get('encoding', 'unknown')
                encoding_counts[enc] = encoding_counts.get(enc, 0) + 1

            most_common_encoding = max(encoding_counts, key=encoding_counts.get)
            recommendations.append(
                f"Consider examining files with encoding errors. Most common failing encoding: "
                f"{most_common_encoding} ({encoding_counts[most_common_encoding]} files)"
            )

        total_processed = self.successful_count + len(self.failed_files)
        if total_processed > 0 and len(self.failed_files) > total_processed * 0.1:  # More than 10% failure rate
            recommendations.append(
                "High failure rate detected. Check data source quality and consider manual inspection of quarantined files."
            )

        if not recommendations:
            recommendations.append("No specific issues detected. Pipeline processed files successfully.")

        return recommendations

    def _save_checkpoint(self, step: str, completed_files: List[str] = None, metadata: dict = None) -> None:
        """Save checkpoint for pipeline progress."""
        if not self.enable_checkpoints:
            return

        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'completed_step': step,
            'completed_files': completed_files or [],
            'successful_count': self.successful_count,
            'failed_files': self.failed_files,
            'metadata': metadata or {}
        }

        try:
            utils.create_dir(os.path.dirname(self.checkpoint_file))
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logging.debug(f"Saved checkpoint for step: {step}")
        except Exception as e:
            logging.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> dict:
        """Load existing checkpoint if available."""
        if not self.enable_checkpoints or not os.path.exists(self.checkpoint_file):
            return {}

        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            logging.info(f"Loaded checkpoint from: {checkpoint_data.get('completed_step', 'unknown')}")
            return checkpoint_data
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
            return {}

    def _is_step_completed(self, step: str) -> bool:
        """Check if a pipeline step was already completed."""
        checkpoint = self._load_checkpoint()
        completed_step = checkpoint.get('completed_step', '')

        # Define step order for comparison
        step_order = [
            'split_fasta',
            'convert_fasta_to_csv',
            'clean_csv_data',
            'sort_into_periods',
            'remove_period_duplicates',
            'remove_empty_periods',
            'create_unique_sequence_files'
        ]

        if completed_step in step_order and step in step_order:
            return step_order.index(completed_step) >= step_order.index(step)

        return False

    def _optimize_csv_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage."""
        for col in df.columns:
            col_type = df[col].dtype

            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
        return df

    def _get_file_line_count(self, file_path: str) -> int:
        """Get the number of lines in a file."""
        cmd = f'wc -l {file_path}'
        output = subprocess.check_output(cmd, shell=True, encoding='UTF-8')
        return int(output.split()[0])
    
    def _remove_temp_fasta(self) -> None:
        """Remove temporary FASTA directory."""
        logging.info("Removing temporary FASTA files...")
        temp_fasta_dir = self.data_config['temp_fasta_dir']
        shutil.rmtree(temp_fasta_dir, ignore_errors=True)
    
    def _remove_temp_csv(self) -> None:
        """Remove temporary CSV directory."""
        logging.info("Removing temporary CSV files...")
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
    
    def _has_parsed_description(self, df: pd.DataFrame) -> bool:
        """Check if description has already been parsed into isolate_name and timestamp columns."""
        return 'isolate_name' in df.columns and 'timestamp' in df.columns

    def _parse_and_filter_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse description field and extract relevant information."""
        # Split description: gene|isolate_name|timestamp|rest
        splitted_desc = df['description'].str.split(pat='|', expand=True, n=3)
        splitted_desc.columns = ['gene', 'isolate_name', 'timestamp', 'rest']

        # Keep only isolate_name and timestamp
        splitted_desc = splitted_desc[['isolate_name', 'timestamp']].copy()

        # Adapt day format (add 1 day) with proper date arithmetic
        try:
            # First, ensure zero-padding for consistent parsing
            fixed_timestamps = self._validate_and_fix_timestamps(splitted_desc['timestamp'])

            # Convert to datetime, add 1 day, then back to string format
            timestamps_dt = pd.to_datetime(fixed_timestamps, format='%Y-%m-%d', errors='coerce')

            # Add 1 day using proper date arithmetic
            timestamps_dt = timestamps_dt + pd.Timedelta(days=1)

            # Convert back to string format
            splitted_desc['timestamp'] = timestamps_dt.dt.strftime('%Y-%m-%d')

            # Handle any NaT values that couldn't be parsed
            invalid_count = splitted_desc['timestamp'].isna().sum()
            if invalid_count > 0:
                logging.warning(f"Could not parse {invalid_count} timestamps in description parsing, keeping original values")
                # For invalid timestamps, keep the original fixed format without adding 1 day
                mask = timestamps_dt.isna()
                splitted_desc.loc[mask, 'timestamp'] = fixed_timestamps[mask]

        except Exception as e:
            logging.warning(f"Error processing timestamps in description parsing: {e}")
            # Keep original timestamps if processing fails
            pass

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