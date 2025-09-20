#!/usr/bin/env python3
"""
Microbiome Data Preprocessing Pipeline for MPEG-G Challenge
Handles large FASTQ files and outputs manageable feature matrices
Author: MPEG-G Challenge Participant
"""

import os
import sys
import gzip
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import warnings
from typing import Dict, List, Tuple, Optional
import subprocess
import multiprocessing as mp
from functools import partial
import pickle
import h5py
from tqdm import tqdm
import hashlib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MicrobiomePreprocessor:
    """
    Main preprocessing class for handling large microbiome datasets
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 metadata_file: str,
                 cytokine_file: Optional[str] = None,
                 chunk_size: int = 10000,
                 n_cores: int = None):
        """
        Initialize preprocessor
        
        Args:
            input_dir: Directory containing FASTQ files
            output_dir: Directory for processed outputs
            metadata_file: Path to metadata CSV (Train.csv)
            cytokine_file: Path to cytokine measurements (if available)
            chunk_size: Number of sequences to process at once
            n_cores: Number of CPU cores to use
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_file = metadata_file
        self.cytokine_file = cytokine_file
        self.chunk_size = chunk_size
        self.n_cores = n_cores or mp.cpu_count() - 1
        
        # Create output directories
        self.create_output_structure()
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        self.subjects_info = pd.read_csv(metadata_file.replace('Train.csv', 'Train_Subjects.csv'))
        
        logger.info(f"Initialized preprocessor with {self.n_cores} cores")
        
    def create_output_structure(self):
        """Create organized output directory structure"""
        dirs = [
            'otu_tables',
            'diversity_metrics', 
            'taxonomic_profiles',
            'feature_matrices',
            'temporal_data',
            'quality_reports',
            'intermediate',
            'visualizations'
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def process_fastq_chunk(self, fastq_file: str, start_idx: int = 0) -> Dict:
        """
        Process a chunk of FASTQ file
        
        Args:
            fastq_file: Path to FASTQ file
            start_idx: Starting index for chunk processing
            
        Returns:
            Dictionary with sequence statistics and quality metrics
        """
        sequences = []
        qualities = []
        headers = []
        
        # Handle gzipped files
        opener = gzip.open if fastq_file.endswith('.gz') else open
        
        try:
            with opener(fastq_file, 'rt') as f:
                # Skip to start position if resuming
                for _ in range(start_idx * 4):
                    next(f, None)
                
                count = 0
                while count < self.chunk_size:
                    header = f.readline().strip()
                    if not header:
                        break
                    
                    seq = f.readline().strip()
                    plus = f.readline().strip()
                    qual = f.readline().strip()
                    
                    if header.startswith('@'):
                        headers.append(header[1:])
                        sequences.append(seq)
                        qualities.append(qual)
                        count += 1
                        
        except Exception as e:
            logger.error(f"Error processing {fastq_file}: {e}")
            return {}
            
        return {
            'headers': headers,
            'sequences': sequences,
            'qualities': qualities,
            'file': os.path.basename(fastq_file),
            'chunk_id': start_idx
        }
    
    def quality_filter_sequences(self, data: Dict, min_quality: int = 20) -> Dict:
        """
        Filter sequences based on quality scores
        
        Args:
            data: Dictionary from process_fastq_chunk
            min_quality: Minimum average quality score
            
        Returns:
            Filtered sequences dictionary
        """
        filtered_seqs = []
        filtered_quals = []
        filtered_headers = []
        
        for seq, qual, header in zip(data['sequences'], data['qualities'], data['headers']):
            # Convert quality string to scores
            qual_scores = [ord(c) - 33 for c in qual]
            avg_quality = np.mean(qual_scores)
            
            if avg_quality >= min_quality:
                filtered_seqs.append(seq)
                filtered_quals.append(qual)
                filtered_headers.append(header)
                
        return {
            'headers': filtered_headers,
            'sequences': filtered_seqs,
            'qualities': filtered_quals,
            'file': data['file'],
            'original_count': len(data['sequences']),
            'filtered_count': len(filtered_seqs)
        }
    
    def create_otu_table(self, sequences_dict: Dict) -> pd.DataFrame:
        """
        Create OTU table from sequences (simplified version)
        In production, use QIIME2 or DADA2
        
        Args:
            sequences_dict: Dictionary of processed sequences
            
        Returns:
            OTU abundance table
        """
        # This is a simplified OTU clustering
        # For real analysis, use proper tools like VSEARCH or USEARCH
        
        otu_dict = {}
        sequence_to_otu = {}
        otu_counter = 0
        
        for sample_id, seqs in sequences_dict.items():
            sample_otus = {}
            
            for seq in seqs['sequences']:
                # Simple hash-based OTU assignment (replace with proper clustering)
                seq_hash = hashlib.md5(seq.encode()).hexdigest()[:8]
                
                if seq_hash not in sequence_to_otu:
                    sequence_to_otu[seq_hash] = f"OTU_{otu_counter:05d}"
                    otu_counter += 1
                
                otu_id = sequence_to_otu[seq_hash]
                sample_otus[otu_id] = sample_otus.get(otu_id, 0) + 1
            
            otu_dict[sample_id] = sample_otus
        
        # Convert to DataFrame
        otu_df = pd.DataFrame.from_dict(otu_dict, orient='index')
        otu_df.fillna(0, inplace=True)
        
        return otu_df
    
    def calculate_diversity_metrics(self, otu_table: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate alpha and beta diversity metrics
        
        Args:
            otu_table: OTU abundance table
            
        Returns:
            DataFrame with diversity metrics
        """
        diversity_metrics = {}
        
        for sample in otu_table.index:
            abundances = otu_table.loc[sample]
            abundances = abundances[abundances > 0]
            
            # Alpha diversity metrics
            metrics = {
                'observed_otus': len(abundances),
                'shannon': self._shannon_index(abundances),
                'simpson': self._simpson_index(abundances),
                'chao1': self._chao1_index(abundances),
                'evenness': self._pielou_evenness(abundances)
            }
            
            diversity_metrics[sample] = metrics
        
        return pd.DataFrame.from_dict(diversity_metrics, orient='index')
    
    def _shannon_index(self, abundances):
        """Calculate Shannon diversity index"""
        proportions = abundances / abundances.sum()
        return -sum(p * np.log(p) for p in proportions if p > 0)
    
    def _simpson_index(self, abundances):
        """Calculate Simpson diversity index"""
        proportions = abundances / abundances.sum()
        return 1 - sum(p**2 for p in proportions)
    
    def _chao1_index(self, abundances):
        """Estimate Chao1 richness"""
        S_obs = len(abundances)
        singletons = sum(abundances == 1)
        doubletons = sum(abundances == 2)
        
        if doubletons > 0:
            return S_obs + (singletons**2) / (2 * doubletons)
        else:
            return S_obs + singletons * (singletons - 1) / 2
    
    def _pielou_evenness(self, abundances):
        """Calculate Pielou's evenness"""
        if len(abundances) <= 1:
            return 0
        return self._shannon_index(abundances) / np.log(len(abundances))
    
    def process_cytokine_data(self, cytokine_file: str = None) -> pd.DataFrame:
        """
        Process cytokine measurements
        
        Args:
            cytokine_file: Path to cytokine data file
            
        Returns:
            Processed cytokine DataFrame
        """
        if cytokine_file is None and self.cytokine_file is None:
            logger.warning("No cytokine file provided")
            return pd.DataFrame()
        
        file_to_use = cytokine_file or self.cytokine_file
        
        try:
            # Assuming cytokine data is in CSV format
            cytokines = pd.read_csv(file_to_use)
            
            # Log-transform cytokine values (common preprocessing)
            numeric_cols = cytokines.select_dtypes(include=[np.number]).columns
            cytokines[numeric_cols] = np.log1p(cytokines[numeric_cols])
            
            # Normalize cytokine values
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            cytokines[numeric_cols] = scaler.fit_transform(cytokines[numeric_cols])
            
            # Save scaler for later use
            import joblib
            joblib.dump(scaler, self.output_dir / 'cytokine_scaler.pkl')
            
            return cytokines
            
        except Exception as e:
            logger.error(f"Error processing cytokine data: {e}")
            return pd.DataFrame()
    
    def create_feature_matrix(self, 
                            otu_table: pd.DataFrame,
                            diversity_metrics: pd.DataFrame,
                            cytokines: pd.DataFrame,
                            metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all features into single matrix
        
        Args:
            otu_table: OTU abundance table
            diversity_metrics: Diversity metrics
            cytokines: Cytokine measurements
            metadata: Sample metadata
            
        Returns:
            Combined feature matrix
        """
        # Start with metadata
        feature_matrix = metadata.set_index('SampleID')
        
        # Add OTU features (select top N most abundant)
        top_otus = otu_table.sum().nlargest(100).index
        otu_features = otu_table[top_otus].add_prefix('OTU_')
        feature_matrix = feature_matrix.join(otu_features, how='left')
        
        # Add diversity metrics
        diversity_features = diversity_metrics.add_prefix('Diversity_')
        feature_matrix = feature_matrix.join(diversity_features, how='left')
        
        # Add cytokine features if available
        if not cytokines.empty:
            cytokine_features = cytokines.set_index('SampleID').add_prefix('Cytokine_')
            feature_matrix = feature_matrix.join(cytokine_features, how='left')
        
        # Add temporal features
        if 'Date' in feature_matrix.columns:
            feature_matrix['Date'] = pd.to_datetime(feature_matrix['Date'])
            feature_matrix['DaysSinceStart'] = (
                feature_matrix['Date'] - feature_matrix['Date'].min()
            ).dt.days
        
        return feature_matrix
    
    def process_batch_parallel(self, fastq_files: List[str]) -> Dict:
        """
        Process multiple FASTQ files in parallel
        
        Args:
            fastq_files: List of FASTQ file paths
            
        Returns:
            Dictionary of processed sequences
        """
        logger.info(f"Processing {len(fastq_files)} files in parallel")
        
        with mp.Pool(self.n_cores) as pool:
            results = pool.map(self.process_fastq_chunk, fastq_files)
        
        # Combine results
        combined = {}
        for result in results:
            if result:
                sample_id = result['file'].replace('.fastq', '').replace('.gz', '')
                combined[sample_id] = result
                
        return combined
    
    def save_processed_data(self, 
                           otu_table: pd.DataFrame,
                           diversity_metrics: pd.DataFrame,
                           feature_matrix: pd.DataFrame,
                           format: str = 'multiple'):
        """
        Save processed data in various formats
        
        Args:
            otu_table: OTU abundance table
            diversity_metrics: Diversity metrics
            feature_matrix: Combined feature matrix
            format: 'csv', 'hdf5', 'pickle', or 'multiple'
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format in ['csv', 'multiple']:
            # Save as CSV (readable format)
            otu_table.to_csv(
                self.output_dir / f'otu_tables/otu_table_{timestamp}.csv'
            )
            diversity_metrics.to_csv(
                self.output_dir / f'diversity_metrics/diversity_{timestamp}.csv'
            )
            feature_matrix.to_csv(
                self.output_dir / f'feature_matrices/features_{timestamp}.csv'
            )
            
        if format in ['hdf5', 'multiple']:
            # Save as HDF5 (efficient for large data)
            with h5py.File(self.output_dir / f'processed_data_{timestamp}.h5', 'w') as f:
                f.create_dataset('otu_table', data=otu_table.values)
                f.create_dataset('diversity', data=diversity_metrics.values)
                f.create_dataset('features', data=feature_matrix.values)
                
                # Save column names
                f.attrs['otu_columns'] = list(otu_table.columns)
                f.attrs['diversity_columns'] = list(diversity_metrics.columns)
                f.attrs['feature_columns'] = list(feature_matrix.columns)
                
        if format in ['pickle', 'multiple']:
            # Save as pickle (preserves all data types)
            with open(self.output_dir / f'processed_data_{timestamp}.pkl', 'wb') as f:
                pickle.dump({
                    'otu_table': otu_table,
                    'diversity_metrics': diversity_metrics,
                    'feature_matrix': feature_matrix
                }, f)
                
        # Create summary report
        self.create_summary_report(otu_table, diversity_metrics, feature_matrix, timestamp)
        
        logger.info(f"Data saved successfully with timestamp {timestamp}")
        
    def create_summary_report(self, 
                             otu_table: pd.DataFrame,
                             diversity_metrics: pd.DataFrame,
                             feature_matrix: pd.DataFrame,
                             timestamp: str):
        """
        Create a summary report of processed data
        
        Args:
            otu_table: OTU abundance table
            diversity_metrics: Diversity metrics
            feature_matrix: Combined feature matrix
            timestamp: Processing timestamp
        """
        report = {
            'processing_date': timestamp,
            'n_samples': len(otu_table),
            'n_otus': len(otu_table.columns),
            'n_features': len(feature_matrix.columns),
            'avg_reads_per_sample': otu_table.sum(axis=1).mean(),
            'avg_otus_per_sample': (otu_table > 0).sum(axis=1).mean(),
            'shannon_diversity_mean': diversity_metrics['shannon'].mean() if 'shannon' in diversity_metrics else None,
            'file_sizes': {
                'otu_table_mb': otu_table.memory_usage(deep=True).sum() / 1024**2,
                'feature_matrix_mb': feature_matrix.memory_usage(deep=True).sum() / 1024**2
            }
        }
        
        with open(self.output_dir / f'quality_reports/summary_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Also create human-readable report
        with open(self.output_dir / f'quality_reports/report_{timestamp}.txt', 'w') as f:
            f.write("="*50 + "\n")
            f.write("PREPROCESSING SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
                
    def run_full_pipeline(self):
        """
        Run the complete preprocessing pipeline
        """
        logger.info("Starting full preprocessing pipeline")
        
        # Step 1: Find all FASTQ files
        fastq_files = list(self.input_dir.glob("*.fastq*"))
        logger.info(f"Found {len(fastq_files)} FASTQ files")
        
        if not fastq_files:
            logger.error("No FASTQ files found!")
            return
        
        # Step 2: Process FASTQ files in batches
        all_sequences = {}
        batch_size = min(10, len(fastq_files))
        
        for i in tqdm(range(0, len(fastq_files), batch_size), desc="Processing batches"):
            batch = fastq_files[i:i+batch_size]
            batch_results = self.process_batch_parallel([str(f) for f in batch])
            all_sequences.update(batch_results)
            
            # Save intermediate results
            if i % 50 == 0:
                with open(self.output_dir / f'intermediate/sequences_batch_{i}.pkl', 'wb') as f:
                    pickle.dump(all_sequences, f)
        
        # Step 3: Quality filtering
        logger.info("Applying quality filters")
        filtered_sequences = {}
        for sample_id, data in all_sequences.items():
            filtered_sequences[sample_id] = self.quality_filter_sequences(data)
        
        # Step 4: Create OTU table
        logger.info("Creating OTU table")
        otu_table = self.create_otu_table(filtered_sequences)
        
        # Step 5: Calculate diversity metrics
        logger.info("Calculating diversity metrics")
        diversity_metrics = self.calculate_diversity_metrics(otu_table)
        
        # Step 6: Process cytokine data if available
        logger.info("Processing cytokine data")
        cytokines = self.process_cytokine_data()
        
        # Step 7: Create feature matrix
        logger.info("Creating feature matrix")
        feature_matrix = self.create_feature_matrix(
            otu_table, diversity_metrics, cytokines, self.metadata
        )
        
        # Step 8: Save all processed data
        logger.info("Saving processed data")
        self.save_processed_data(otu_table, diversity_metrics, feature_matrix)
        
        logger.info("Pipeline completed successfully!")
        
        return {
            'otu_table': otu_table,
            'diversity_metrics': diversity_metrics,
            'feature_matrix': feature_matrix,
            'summary': {
                'n_samples': len(otu_table),
                'n_features': len(feature_matrix.columns),
                'output_dir': str(self.output_dir)
            }
        }

def main():
    """
    Main execution function
    """
    # Configuration
    config = {
        'input_dir': '/path/to/your/fastq/files',  # UPDATE THIS
        'output_dir': './processed_data',
        'metadata_file': './Train.csv',  # UPDATE THIS
        'cytokine_file': './cytokine_measurements.csv',  # UPDATE THIS if available
        'chunk_size': 10000,
        'n_cores': mp.cpu_count() - 1
    }
    
    # Initialize preprocessor
    preprocessor = MicrobiomePreprocessor(**config)
    
    # Run pipeline
    results = preprocessor.run_full_pipeline()
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Output files saved to: {config['output_dir']}")
    print(f"Total samples processed: {results['summary']['n_samples']}")
    print(f"Total features generated: {results['summary']['n_features']}")
    print("\nGenerated output files:")
    print("1. processed_data/feature_matrices/features_*.csv")
    print("2. processed_data/diversity_metrics/diversity_*.csv")
    print("3. processed_data/otu_tables/otu_table_*.csv (if small enough)")
    print("4. processed_data/quality_reports/summary_*.json")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Microbiome Data Preprocessing Pipeline')
    parser.add_argument('--input-dir', required=True, help='Directory containing FASTQ files')
    parser.add_argument('--output-dir', default='./processed_data', help='Output directory')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV file')
    parser.add_argument('--cytokines', help='Path to cytokine measurements file')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for processing')
    parser.add_argument('--cores', type=int, help='Number of CPU cores to use')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'metadata_file': args.metadata,
        'cytokine_file': args.cytokines,
        'chunk_size': args.chunk_size,
        'n_cores': args.cores or mp.cpu_count() - 1
    }
    
    # Initialize and run
    preprocessor = MicrobiomePreprocessor(**config)
    results = preprocessor.run_full_pipeline()
    
    print(f"\nProcessing complete! Check {args.output_dir} for results.")
    
    # Print summary for pipeline integration
    print(f"Summary: Processed {results.get('summary', {}).get('n_samples', 'unknown')} samples")
    print(f"Generated {results.get('summary', {}).get('n_features', 'unknown')} features")
    print("Preprocessing pipeline completed successfully!")