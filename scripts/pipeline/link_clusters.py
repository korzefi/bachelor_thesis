#!/usr/bin/env python3
"""
Cluster Linking Pipeline Module

This module handles the linking of clusters across consecutive time periods by:
1. Loading cluster centroids from all periods
2. Computing Euclidean distances between centroids
3. Creating bidirectional links between the closest clusters
4. Saving the linked cluster information

Refactored from: scripts/preprocessing/ClusterLinker.py
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from natsort import natsort_keygen
from typing import Dict, List, Tuple


class ClusterLinkingPipeline:
    """Main class that orchestrates the cluster linking process."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration dictionary."""
        self.config = config
        self.data_config = config['data']
        
        # Setup paths
        self.centroids_file = self.data_config['centroids_file']
        self.linked_centroids_file = self.data_config['linked_centroids_file']
    
    def run(self) -> None:
        """Execute the complete cluster linking pipeline."""
        logging.info("Starting cluster linking pipeline...")
        
        try:
            # Load centroids data
            centroids_df = self._load_centroids_data()
            
            # Sort centroids by period and cluster
            sorted_df = self._sort_centroids(centroids_df)
            
            # Create links between consecutive periods
            next_cluster_links = self._create_cluster_links(sorted_df)
            
            # Add links to the dataframe
            self._add_links_to_dataframe(sorted_df, next_cluster_links)
            
            # Save linked centroids
            self._save_linked_centroids(sorted_df)
            
            logging.info("Cluster linking pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Cluster linking pipeline failed: {e}")
            raise
    
    def _load_centroids_data(self) -> pd.DataFrame:
        """Load cluster centroids data from CSV file."""
        if not Path(self.centroids_file).exists():
            raise FileNotFoundError(f"Centroids file not found: {self.centroids_file}")
        
        logging.info(f"Loading centroids data from: {self.centroids_file}")
        df = pd.read_csv(self.centroids_file)
        
        if df.empty:
            raise ValueError("Centroids file is empty")
        
        logging.info(f"Loaded {len(df)} cluster centroids from {df['period'].nunique()} periods")
        return df
    
    def _sort_centroids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort centroids by period and cluster number using natural sorting."""
        logging.info("Sorting centroids by period and cluster...")
        df_sorted = df.copy()
        df_sorted.sort_values(
            by=['period', 'cluster'], 
            key=natsort_keygen(), 
            inplace=True
        )
        df_sorted.reset_index(drop=True, inplace=True)
        return df_sorted
    
    def _create_cluster_links(self, df: pd.DataFrame) -> List[str]:
        """Create links between clusters in consecutive periods."""
        logging.info("Creating cluster links between consecutive periods...")
        
        periods = df['period'].unique()
        
        if len(periods) < 2:
            raise ValueError("Need at least 2 periods to create cluster links")
        
        logging.info(f"Linking clusters across {len(periods)} periods: {periods}")
        
        # Get 100-dimensional feature columns
        feature_columns = [f'd{i}' for i in range(1, 101)]
        
        # Initialize link dictionaries
        forward_links = {}  # {current_period_row_idx: next_period_row_idx}
        backward_links = {}  # {next_period_row_idx: current_period_row_idx}
        
        # Create links between consecutive periods
        for i in range(len(periods) - 1):
            current_period = periods[i]
            next_period = periods[i + 1]
            
            logging.info(f"Linking {current_period} -> {next_period}")
            
            # Forward links: for each cluster in current period, find closest in next period
            forward_links.update(
                self._create_forward_links(df, current_period, next_period, feature_columns)
            )
            
            # Backward links: for each cluster in next period, find closest in current period
            backward_links.update(
                self._create_backward_links(df, current_period, next_period, feature_columns)
            )
        
        # Combine and format links
        combined_links = self._combine_bidirectional_links(forward_links, backward_links)
        cluster_link_strings = self._format_cluster_links(df, combined_links)
        
        return cluster_link_strings
    
    def _create_forward_links(
        self, 
        df: pd.DataFrame, 
        current_period: str, 
        next_period: str, 
        feature_columns: List[str]
    ) -> Dict[int, int]:
        """Create forward links from current period to next period."""
        links = {}
        
        current_clusters = df[df['period'] == current_period]
        next_clusters = df[df['period'] == next_period]
        
        for current_idx, current_row in current_clusters.iterrows():
            min_distance = float('inf')
            closest_idx = None
            
            for next_idx, next_row in next_clusters.iterrows():
                distance = self._calculate_euclidean_distance(
                    current_row[feature_columns], 
                    next_row[feature_columns]
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = next_idx
            
            if closest_idx is not None:
                links[current_idx] = closest_idx
        
        return links
    
    def _create_backward_links(
        self, 
        df: pd.DataFrame, 
        current_period: str, 
        next_period: str, 
        feature_columns: List[str]
    ) -> Dict[int, int]:
        """Create backward links from next period to current period."""
        links = {}
        
        current_clusters = df[df['period'] == current_period]
        next_clusters = df[df['period'] == next_period]
        
        for next_idx, next_row in next_clusters.iterrows():
            min_distance = float('inf')
            closest_idx = None
            
            for current_idx, current_row in current_clusters.iterrows():
                distance = self._calculate_euclidean_distance(
                    current_row[feature_columns], 
                    next_row[feature_columns]
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = current_idx
            
            if closest_idx is not None:
                links[next_idx] = closest_idx
        
        return links
    
    def _calculate_euclidean_distance(
        self, 
        vector1: pd.Series, 
        vector2: pd.Series
    ) -> float:
        """Calculate Euclidean distance between two vectors."""
        vec1 = np.array(vector1)
        vec2 = np.array(vector2)
        return np.linalg.norm(vec1 - vec2)
    
    def _combine_bidirectional_links(
        self, 
        forward_links: Dict[int, int], 
        backward_links: Dict[int, int]
    ) -> Dict[int, List[int]]:
        """Combine forward and backward links into bidirectional links."""
        combined_links = {}
        
        # Convert forward links to lists
        for current_idx, next_idx in forward_links.items():
            if current_idx not in combined_links:
                combined_links[current_idx] = []
            combined_links[current_idx].append(next_idx)
        
        # Reverse backward links and add them
        reversed_backward = {}
        for next_idx, current_idx in backward_links.items():
            if current_idx not in reversed_backward:
                reversed_backward[current_idx] = []
            reversed_backward[current_idx].append(next_idx)
        
        # Merge reversed backward links
        for current_idx, next_indices in reversed_backward.items():
            if current_idx not in combined_links:
                combined_links[current_idx] = []
            combined_links[current_idx].extend(next_indices)
        
        # Remove duplicates and sort
        for idx in combined_links:
            combined_links[idx] = sorted(list(set(combined_links[idx])))
        
        return combined_links
    
    def _format_cluster_links(
        self, 
        df: pd.DataFrame, 
        combined_links: Dict[int, List[int]]
    ) -> List[str]:
        """Format cluster links as strings."""
        cluster_link_strings = []
        
        for i in range(len(df)):
            if i in combined_links:
                # Get cluster numbers for the linked indices
                linked_clusters = [df.loc[idx, 'cluster'] for idx in combined_links[i]]
                # Convert to string format: "cluster1-cluster2-cluster3"
                link_string = '-'.join(map(str, linked_clusters))
            else:
                # No links for this cluster
                link_string = ''
            
            cluster_link_strings.append(link_string)
        
        return cluster_link_strings
    
    def _add_links_to_dataframe(
        self, 
        df: pd.DataFrame, 
        cluster_links: List[str]
    ) -> None:
        """Add cluster links column to the dataframe."""
        logging.info("Adding cluster links to dataframe...")
        
        # Remove existing next_cluster column if it exists
        if 'next_cluster' in df.columns:
            df.drop('next_cluster', axis=1, inplace=True)
        
        # Insert next_cluster column after cluster column
        df.insert(2, 'next_cluster', cluster_links)
    
    def _save_linked_centroids(self, df: pd.DataFrame) -> None:
        """Save the linked centroids to file."""
        logging.info(f"Saving linked centroids to: {self.linked_centroids_file}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.linked_centroids_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        df.to_csv(self.linked_centroids_file, index=False)
        
        logging.info(f"Saved {len(df)} linked cluster centroids")


def run(config: Dict) -> None:
    """Main entry point for the cluster linking pipeline."""
    pipeline = ClusterLinkingPipeline(config)
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