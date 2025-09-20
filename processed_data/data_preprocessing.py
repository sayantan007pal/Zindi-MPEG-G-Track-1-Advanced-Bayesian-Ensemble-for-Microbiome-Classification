#!/usr/bin/env python3
"""
MPEG-G Microbiome Challenge Track 1: Data Preprocessing Pipeline
==============================================================

This script handles missing values, normalization, feature selection,
and creates ML-ready datasets for cytokine prediction.

Author: Data Processing Pipeline
Date: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MPEGDataPreprocessor:
    """Class to preprocess MPEG-G microbiome data for machine learning"""
    
    def __init__(self, processed_data_path):
        self.processed_data_path = Path(processed_data_path)
        self.feature_matrix = None
        self.cytokine_targets = None
        self.processed_features = None
        self.processed_targets = None
        self.feature_names = None
        self.target_names = None
        self.preprocessing_stats = {}
        
    def load_processed_data(self):
        """Load the processed feature and target matrices"""
        print("Loading processed data...")
        
        # Load feature matrix
        feature_path = self.processed_data_path / "feature_matrix.csv"
        if feature_path.exists():
            self.feature_matrix = pd.read_csv(feature_path, index_col=0)
            print(f"✓ Feature matrix loaded: {self.feature_matrix.shape}")
        else:
            raise FileNotFoundError(f"Feature matrix not found at {feature_path}")
            
        # Load cytokine targets
        target_path = self.processed_data_path / "cytokine_targets.csv"
        if target_path.exists():
            self.cytokine_targets = pd.read_csv(target_path, index_col=0)
            print(f"✓ Cytokine targets loaded: {self.cytokine_targets.shape}")
        else:
            raise FileNotFoundError(f"Cytokine targets not found at {target_path}")
            
        # Store original names
        self.feature_names = list(self.feature_matrix.columns)
        self.target_names = list(self.cytokine_targets.columns)
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Targets: {len(self.target_names)}")
        
    def handle_missing_values(self, strategy='median'):
        """Handle missing values in features and targets"""
        print(f"\nHandling missing values with strategy: {strategy}")
        
        # Check missing values in features
        feature_missing = self.feature_matrix.isnull().sum()
        missing_features = feature_missing[feature_missing > 0]
        
        if len(missing_features) > 0:
            print(f"Features with missing values: {len(missing_features)}")
            print(f"Max missing in features: {missing_features.max()}")
            
            # Impute missing values
            if strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=strategy)
                self.feature_matrix = pd.DataFrame(
                    imputer.fit_transform(self.feature_matrix),
                    index=self.feature_matrix.index,
                    columns=self.feature_matrix.columns
                )
            elif strategy == 'zero':
                self.feature_matrix = self.feature_matrix.fillna(0)
                
            print(f"✓ Missing values in features handled")
        else:
            print("✓ No missing values in features")
            
        # Check missing values in targets
        target_missing = self.cytokine_targets.isnull().sum()
        missing_targets = target_missing[target_missing > 0]
        
        if len(missing_targets) > 0:
            print(f"Targets with missing values: {len(missing_targets)}")
            
            # For targets, use median imputation
            target_imputer = SimpleImputer(strategy='median')
            self.cytokine_targets = pd.DataFrame(
                target_imputer.fit_transform(self.cytokine_targets),
                index=self.cytokine_targets.index,
                columns=self.cytokine_targets.columns
            )
            print(f"✓ Missing values in targets handled")
        else:
            print("✓ No missing values in targets")
            
        self.preprocessing_stats['missing_values_handled'] = True
        
    def normalize_features(self, method='relative_abundance'):
        """Normalize microbiome features"""
        print(f"\nNormalizing features with method: {method}")
        
        if method == 'relative_abundance':
            # Convert to relative abundance (divide by total reads per sample)
            row_sums = self.feature_matrix.sum(axis=1)
            self.processed_features = self.feature_matrix.div(row_sums, axis=0)
            
            # Replace any NaN values with 0 (in case of samples with zero total reads)
            self.processed_features = self.processed_features.fillna(0)
            
            print(f"✓ Converted to relative abundance")
            
        elif method == 'log_transform':
            # Log transformation: log(x + 1)
            self.processed_features = np.log1p(self.feature_matrix)
            print(f"✓ Applied log transformation")
            
        elif method == 'relative_log':
            # Relative abundance + log transformation
            row_sums = self.feature_matrix.sum(axis=1)
            relative_abundance = self.feature_matrix.div(row_sums, axis=0).fillna(0)
            self.processed_features = np.log1p(relative_abundance)
            print(f"✓ Applied relative abundance + log transformation")
            
        elif method == 'standardize':
            # Standard scaling (z-score normalization)
            scaler = StandardScaler()
            self.processed_features = pd.DataFrame(
                scaler.fit_transform(self.feature_matrix),
                index=self.feature_matrix.index,
                columns=self.feature_matrix.columns
            )
            print(f"✓ Applied standard scaling")
            
        elif method == 'minmax':
            # Min-max scaling
            scaler = MinMaxScaler()
            self.processed_features = pd.DataFrame(
                scaler.fit_transform(self.feature_matrix),
                index=self.feature_matrix.index,
                columns=self.feature_matrix.columns
            )
            print(f"✓ Applied min-max scaling")
        else:
            # No normalization
            self.processed_features = self.feature_matrix.copy()
            print(f"✓ No normalization applied")
            
        self.preprocessing_stats['normalization_method'] = method
        
    def filter_features(self, min_prevalence=0.1, variance_threshold=0.0):
        """Filter features based on prevalence and variance"""
        print(f"\nFiltering features...")
        print(f"Original features: {self.processed_features.shape[1]}")
        
        # Filter by prevalence (fraction of samples where feature is present)
        if min_prevalence > 0:
            prevalence = (self.processed_features > 0).mean()
            high_prevalence_features = prevalence[prevalence >= min_prevalence].index
            
            self.processed_features = self.processed_features[high_prevalence_features]
            print(f"After prevalence filter (>={min_prevalence}): {self.processed_features.shape[1]}")
            
        # Filter by variance
        if variance_threshold > 0:
            selector = VarianceThreshold(threshold=variance_threshold)
            selected_features = selector.fit_transform(self.processed_features)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_feature_names = self.processed_features.columns[selected_mask]
            
            self.processed_features = pd.DataFrame(
                selected_features,
                index=self.processed_features.index,
                columns=selected_feature_names
            )
            print(f"After variance filter (>={variance_threshold}): {self.processed_features.shape[1]}")
            
        self.preprocessing_stats['features_after_filtering'] = self.processed_features.shape[1]
        self.preprocessing_stats['min_prevalence'] = min_prevalence
        self.preprocessing_stats['variance_threshold'] = variance_threshold
        
    def select_top_features(self, k=1000, method='f_regression'):
        """Select top k features using statistical tests"""
        print(f"\nSelecting top {k} features using {method}...")
        
        if k >= self.processed_features.shape[1]:
            print(f"k ({k}) >= number of features ({self.processed_features.shape[1]}), skipping selection")
            return
            
        # For each cytokine, select features and take union
        all_selected_features = set()
        
        for cytokine in self.cytokine_targets.columns:
            y = self.cytokine_targets[cytokine]
            
            if method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=min(k, self.processed_features.shape[1]))
            else:
                raise ValueError(f"Unknown selection method: {method}")
                
            selector.fit(self.processed_features, y)
            selected_mask = selector.get_support()
            selected_features = set(self.processed_features.columns[selected_mask])
            all_selected_features.update(selected_features)
            
        # Keep union of all selected features
        all_selected_features = list(all_selected_features)
        self.processed_features = self.processed_features[all_selected_features]
        
        print(f"✓ Selected {len(all_selected_features)} unique features across all cytokines")
        self.preprocessing_stats['selected_features'] = len(all_selected_features)
        self.preprocessing_stats['selection_method'] = method
        
    def apply_pca(self, n_components=0.95, max_components=500):
        """Apply PCA for dimensionality reduction"""
        print(f"\nApplying PCA...")
        
        # Determine number of components
        if isinstance(n_components, float) and n_components < 1.0:
            # Explained variance ratio
            pca_temp = PCA()
            pca_temp.fit(self.processed_features)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_comp = np.argmax(cumsum_var >= n_components) + 1
            n_comp = min(n_comp, max_components, self.processed_features.shape[1])
        else:
            n_comp = min(int(n_components), max_components, self.processed_features.shape[1])
            
        print(f"Using {n_comp} components")
        
        # Apply PCA
        pca = PCA(n_components=n_comp)
        pca_features = pca.fit_transform(self.processed_features)
        
        # Create new feature names
        pca_feature_names = [f'PC_{i+1}' for i in range(n_comp)]
        
        self.processed_features = pd.DataFrame(
            pca_features,
            index=self.processed_features.index,
            columns=pca_feature_names
        )
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"✓ PCA applied: {n_comp} components explaining {explained_var:.3f} of variance")
        
        self.preprocessing_stats['pca_components'] = n_comp
        self.preprocessing_stats['pca_explained_variance'] = explained_var
        
    def normalize_targets(self, method='standardize'):
        """Normalize cytokine targets"""
        print(f"\nNormalizing targets with method: {method}")
        
        if method == 'standardize':
            scaler = StandardScaler()
            self.processed_targets = pd.DataFrame(
                scaler.fit_transform(self.cytokine_targets),
                index=self.cytokine_targets.index,
                columns=self.cytokine_targets.columns
            )
            print(f"✓ Applied standard scaling to targets")
            
        elif method == 'minmax':
            scaler = MinMaxScaler()
            self.processed_targets = pd.DataFrame(
                scaler.fit_transform(self.cytokine_targets),
                index=self.cytokine_targets.index,
                columns=self.cytokine_targets.columns
            )
            print(f"✓ Applied min-max scaling to targets")
            
        elif method == 'log_transform':
            # Log transformation (add small constant to handle zeros)
            self.processed_targets = np.log1p(self.cytokine_targets)
            print(f"✓ Applied log transformation to targets")
            
        else:
            self.processed_targets = self.cytokine_targets.copy()
            print(f"✓ No normalization applied to targets")
            
        self.preprocessing_stats['target_normalization'] = method
        
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """Create train/test split"""
        from sklearn.model_selection import train_test_split
        
        print(f"\nCreating train/test split (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.processed_features,
            self.processed_targets,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # Can't stratify continuous targets
        )
        
        print(f"✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        # Save splits
        X_train.to_csv(self.processed_data_path / "X_train.csv")
        X_test.to_csv(self.processed_data_path / "X_test.csv")
        y_train.to_csv(self.processed_data_path / "y_train.csv")
        y_test.to_csv(self.processed_data_path / "y_test.csv")
        
        print(f"✓ Train/test splits saved")
        
        self.preprocessing_stats['train_samples'] = X_train.shape[0]
        self.preprocessing_stats['test_samples'] = X_test.shape[0]
        self.preprocessing_stats['test_size'] = test_size
        
        return X_train, X_test, y_train, y_test
        
    def save_processed_data(self):
        """Save all processed data"""
        print(f"\nSaving processed data...")
        
        # Save processed features and targets
        if self.processed_features is not None:
            self.processed_features.to_csv(self.processed_data_path / "processed_features.csv")
            print(f"✓ Processed features saved: {self.processed_features.shape}")
            
        if self.processed_targets is not None:
            self.processed_targets.to_csv(self.processed_data_path / "processed_targets.csv")
            print(f"✓ Processed targets saved: {self.processed_targets.shape}")
            
        # Save preprocessing statistics
        stats_df = pd.DataFrame([self.preprocessing_stats]).T
        stats_df.columns = ['Value']
        stats_df.to_csv(self.processed_data_path / "preprocessing_stats.csv")
        print(f"✓ Preprocessing statistics saved")
        
    def generate_preprocessing_report(self):
        """Generate preprocessing report"""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY REPORT")
        print("="*60)
        
        report = []
        report.append("MPEG-G Microbiome Challenge - Data Preprocessing Summary")
        report.append("=" * 55)
        report.append("")
        
        # Original data info
        report.append("ORIGINAL DATA:")
        report.append(f"• Features: {len(self.feature_names)}")
        report.append(f"• Targets: {len(self.target_names)}")
        report.append(f"• Samples: {self.feature_matrix.shape[0]}")
        report.append("")
        
        # Preprocessing steps
        report.append("PREPROCESSING APPLIED:")
        if 'missing_values_handled' in self.preprocessing_stats:
            report.append("• ✓ Missing values handled")
            
        if 'normalization_method' in self.preprocessing_stats:
            method = self.preprocessing_stats['normalization_method']
            report.append(f"• ✓ Feature normalization: {method}")
            
        if 'features_after_filtering' in self.preprocessing_stats:
            n_features = self.preprocessing_stats['features_after_filtering']
            min_prev = self.preprocessing_stats.get('min_prevalence', 0)
            var_thresh = self.preprocessing_stats.get('variance_threshold', 0)
            report.append(f"• ✓ Feature filtering: {n_features} features (prevalence>={min_prev}, variance>={var_thresh})")
            
        if 'selected_features' in self.preprocessing_stats:
            n_selected = self.preprocessing_stats['selected_features']
            method = self.preprocessing_stats['selection_method']
            report.append(f"• ✓ Feature selection: {n_selected} features ({method})")
            
        if 'pca_components' in self.preprocessing_stats:
            n_comp = self.preprocessing_stats['pca_components']
            var_exp = self.preprocessing_stats['pca_explained_variance']
            report.append(f"• ✓ PCA: {n_comp} components ({var_exp:.3f} variance explained)")
            
        if 'target_normalization' in self.preprocessing_stats:
            method = self.preprocessing_stats['target_normalization']
            report.append(f"• ✓ Target normalization: {method}")
            
        report.append("")
        
        # Final data dimensions
        if self.processed_features is not None:
            report.append("FINAL PROCESSED DATA:")
            report.append(f"• Features: {self.processed_features.shape[1]}")
            report.append(f"• Targets: {self.processed_targets.shape[1] if self.processed_targets is not None else len(self.target_names)}")
            report.append(f"• Samples: {self.processed_features.shape[0]}")
            
        if 'train_samples' in self.preprocessing_stats:
            report.append(f"• Train samples: {self.preprocessing_stats['train_samples']}")
            report.append(f"• Test samples: {self.preprocessing_stats['test_samples']}")
            
        report.append("")
        
        # Files generated
        report.append("OUTPUT FILES:")
        report.append("• processed_features.csv - Processed feature matrix")
        report.append("• processed_targets.csv - Processed target matrix")
        if 'train_samples' in self.preprocessing_stats:
            report.append("• X_train.csv, X_test.csv - Feature train/test splits")
            report.append("• y_train.csv, y_test.csv - Target train/test splits")
        report.append("• preprocessing_stats.csv - Preprocessing statistics")
        
        # Save and print report
        report_text = "\n".join(report)
        
        with open(self.processed_data_path / "preprocessing_report.txt", 'w') as f:
            f.write(report_text)
            
        print(report_text)
        print(f"\nReport saved to: {self.processed_data_path / 'preprocessing_report.txt'}")
        
    def run_preprocessing_pipeline(self, 
                                 missing_strategy='median',
                                 normalization_method='relative_log',
                                 min_prevalence=0.1,
                                 variance_threshold=0.0,
                                 feature_selection_k=1000,
                                 apply_pca=False,
                                 pca_components=0.95,
                                 target_normalization='standardize',
                                 create_splits=True,
                                 test_size=0.2):
        """Run the complete preprocessing pipeline"""
        
        print("Starting MPEG-G Data Preprocessing Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_processed_data()
        
        # Handle missing values
        self.handle_missing_values(strategy=missing_strategy)
        
        # Normalize features
        self.normalize_features(method=normalization_method)
        
        # Filter features
        self.filter_features(min_prevalence=min_prevalence, 
                           variance_threshold=variance_threshold)
        
        # Feature selection
        if feature_selection_k and feature_selection_k < self.processed_features.shape[1]:
            self.select_top_features(k=feature_selection_k)
            
        # Apply PCA if requested
        if apply_pca:
            self.apply_pca(n_components=pca_components)
            
        # Normalize targets
        self.normalize_targets(method=target_normalization)
        
        # Create train/test splits
        if create_splits:
            self.create_train_test_split(test_size=test_size)
            
        # Save processed data
        self.save_processed_data()
        
        # Generate report
        self.generate_preprocessing_report()
        
        print("\n" + "="*50)
        print("Preprocessing pipeline completed successfully!")
        print("="*50)

if __name__ == "__main__":
    # Run preprocessing with default parameters
    processed_data_path = "/Users/sayantanpal100/Desktop/MPEG-G_ Decoding the Dialogue/processed_data"
    
    preprocessor = MPEGDataPreprocessor(processed_data_path)
    preprocessor.run_preprocessing_pipeline()