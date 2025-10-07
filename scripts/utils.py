# author: Filip Korzeniewski


import os
import gc
import psutil
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, Any, Optional, Dict
from contextlib import contextmanager


def get_root_path():
    root_path = os.path.dirname(os.path.abspath(__file__))
    splitted = root_path.split('/')[:-1]
    return '/'.join(splitted)


def create_dir(path, log_level='debug'):
    """
    Create a directory if it doesn't exist.

    Args:
        path: Directory path to create
        log_level: Logging level ('debug', 'info', 'warning')
    """
    path_obj = Path(path)

    if path_obj.exists():
        if path_obj.is_dir():
            if log_level == 'debug':
                logging.debug(f"✓ Directory already exists: {path}")
            elif log_level == 'info':
                logging.info(f"✓ Using existing directory: {path}")
            elif log_level == 'warning':
                logging.warning(f"Directory already exists: {path}")
        else:
            logging.error(f"✗ Path exists but is not a directory: {path}")
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")
    else:
        try:
            os.makedirs(path, exist_ok=True)
            if log_level in ('info', 'warning'):
                logging.info(f"✓ Created directory: {path}")
            else:
                logging.debug(f"✓ Created directory: {path}")
        except OSError as e:
            logging.error(f"✗ Failed to create directory {path}: {e}")
            raise



def setup_logger(process_id=None, date=True, time=True, verbose=False):
    datefmt = "%Y-%m-%d" if date else ""
    timefmt = "%H:%M:%S" if time else ""
    datetimefmt = ""
    if date and time:
        datetimefmt = f"{datefmt} {timefmt}"
    else:
        datetimefmt = f"{datefmt}{timefmt}"

    if process_id is not None:
        process_id = f"_{process_id + 1}"
    else:
        process_id = ""

    # Set log level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(level=log_level,
                        format=f"%(levelname)s{process_id} %(asctime)s: %(message)s",
                        datefmt=datetimefmt)


def get_time_string(time):
    """
    Creates a string representation of minutes and seconds from the given time.
    """
    mins = time // 60
    secs = time % 60
    time_string = ''

    if mins < 10:
        time_string += '  '
    elif mins < 100:
        time_string += ' '

    time_string += '%dm ' % mins

    if secs < 10:
        time_string += ' '

    time_string += '%ds' % secs

    return time_string


@contextmanager
def memory_managed_operation(operation_name: str, threshold_mb: float = 100):
    """Context manager for memory-intensive operations."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024

    logging.debug(f"Starting {operation_name}: {initial_memory:.1f}MB")

    try:
        yield
    finally:
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        delta = final_memory - initial_memory

        if delta > threshold_mb:
            logging.warning(f"{operation_name} used {delta:.1f}MB memory")
            # Force aggressive cleanup
            gc.collect(2)  # Full collection


class MemoryMonitor:
    """Monitor and manage memory usage during processing."""

    def __init__(self, max_memory_mb: Optional[int] = None, gc_frequency: int = 100):
        self.max_memory_mb = max_memory_mb
        self.gc_frequency = gc_frequency
        self.batch_count = 0
        self.process = psutil.Process()
        self.memory_history = []
        self.peak_memory = 0

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        usage = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, usage)
        self.memory_history.append(usage)

        # Keep only last 100 measurements
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]

        return usage

    def predict_memory_usage(self, batches_ahead: int) -> float:
        """Predict future memory usage based on trend."""
        if len(self.memory_history) < 2:
            return self.get_memory_usage_mb()

        # Calculate trend
        recent = self.memory_history[-10:]
        trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0

        return self.get_memory_usage_mb() + (trend * batches_ahead)

    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit."""
        if self.max_memory_mb is None:
            return False
        return self.get_memory_usage_mb() > self.max_memory_mb

    def force_garbage_collection(self) -> None:
        """Force garbage collection with multiple generations."""
        initial_memory = self.get_memory_usage_mb()

        # Collect all generations
        for generation in range(3):
            gc.collect(generation)

        # Clear caches
        if hasattr(gc, 'freeze'):
            gc.freeze()  # Freeze tracked objects
            gc.collect()
            gc.unfreeze()

        final_memory = self.get_memory_usage_mb()
        freed_mb = initial_memory - final_memory

        if freed_mb > 10:
            logging.info(f"GC freed {freed_mb:.1f}MB (peak: {self.peak_memory:.1f}MB)")

    def batch_completed(self) -> None:
        """Call this after each batch completion."""
        self.batch_count += 1

        # Periodic garbage collection
        if self.batch_count % self.gc_frequency == 0:
            self.force_garbage_collection()

        # Memory limit check
        if self.check_memory_limit():
            current_memory = self.get_memory_usage_mb()
            logging.warning(f"Memory usage ({current_memory:.1f}MB) exceeds limit ({self.max_memory_mb}MB)")
            self.force_garbage_collection()


class BatchProcessor(ABC):
    """Abstract base class for batch processing operations."""
    
    def __init__(self, config: Dict, batch_size: int, component_name: str):
        self.config = config
        self.batch_size = batch_size
        self.component_name = component_name
        
        # Initialize memory monitor
        memory_config = config.get('optimization', {}).get('memory', {})
        max_memory = memory_config.get('max_memory_mb')
        gc_frequency = memory_config.get('gc_frequency', 100)
        self.memory_monitor = MemoryMonitor(max_memory, gc_frequency)
        
        # Configuration flags
        self.use_streaming = memory_config.get('use_streaming', True)
        self.intermediate_saves = memory_config.get('intermediate_saves', True)
    
    @abstractmethod
    def process_batch(self, batch_data: Any) -> Any:
        """Process a single batch of data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def save_batch_result(self, result: Any, batch_index: int) -> None:
        """Save batch result. Must be implemented by subclasses."""
        pass
    
    def create_data_iterator(self, data_source: Any) -> Iterator[Any]:
        """Create an iterator for batched data processing."""
        if isinstance(data_source, str) and data_source.endswith('.csv'):
            # CSV file - use pandas chunking
            return pd.read_csv(data_source, chunksize=self.batch_size)
        elif isinstance(data_source, (list, tuple)):
            # List/tuple - create batches
            for i in range(0, len(data_source), self.batch_size):
                yield data_source[i:i + self.batch_size]
        else:
            # Try to iterate directly
            batch = []
            for item in data_source:
                batch.append(item)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            if batch:  # Yield remaining items
                yield batch
    
    def process_in_batches(self, data_source: Any, description: str = "Processing") -> list:
        """Process data source in batches with memory monitoring."""
        logging.info(f"{self.component_name}: Starting {description} with batch size {self.batch_size}")
        
        results = []
        batch_count = 0
        
        try:
            for batch_index, batch_data in enumerate(self.create_data_iterator(data_source)):
                # Log progress
                if batch_index % 10 == 0:
                    memory_mb = self.memory_monitor.get_memory_usage_mb()
                    logging.info(f"{self.component_name}: Processing batch {batch_index + 1}, Memory: {memory_mb:.1f}MB")
                
                # Process batch
                batch_result = self.process_batch(batch_data)
                
                # Save result if configured
                if self.intermediate_saves:
                    self.save_batch_result(batch_result, batch_index)
                else:
                    results.append(batch_result)
                
                # Memory monitoring
                self.memory_monitor.batch_completed()
                batch_count += 1
        
        except Exception as e:
            logging.error(f"{self.component_name}: Batch processing failed at batch {batch_count}: {e}")
            raise
        
        logging.info(f"{self.component_name}: Completed {description}, processed {batch_count} batches")
        return results


class DataFrameChunker:
    """Utility for chunked DataFrame operations."""
    
    @staticmethod
    def read_csv_in_chunks(file_path: str, chunk_size: int, **kwargs) -> Iterator[pd.DataFrame]:
        """Read CSV file in chunks."""
        return pd.read_csv(file_path, chunksize=chunk_size, **kwargs)
    
    @staticmethod
    def write_csv_incrementally(df_chunk: pd.DataFrame, output_path: str, 
                               header: bool = None, mode: str = 'a') -> None:
        """Write DataFrame chunk to CSV incrementally."""
        # Auto-determine header based on file existence
        if header is None:
            header = not os.path.exists(output_path)
        
        df_chunk.to_csv(output_path, mode=mode, header=header, index=False)
    
    @staticmethod
    def merge_csv_files(input_files: list, output_file: str, remove_input: bool = True) -> None:
        """Merge multiple CSV files into one."""
        logging.info(f"Merging {len(input_files)} CSV files into {output_file}")
        
        with open(output_file, 'w') as outfile:
            header_written = False
            
            for file_path in input_files:
                with open(file_path, 'r') as infile:
                    lines = infile.readlines()
                    
                    if not header_written:
                        # Write header from first file
                        outfile.writelines(lines)
                        header_written = True
                    else:
                        # Skip header for subsequent files
                        outfile.writelines(lines[1:])
                
                # Remove input file if requested
                if remove_input:
                    os.remove(file_path)
        
        logging.info(f"CSV merge completed: {output_file}")


