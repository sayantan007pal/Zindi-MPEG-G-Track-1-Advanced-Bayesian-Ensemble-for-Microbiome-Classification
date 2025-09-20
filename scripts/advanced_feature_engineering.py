#!/usr/bin/env python3
"""
Advanced Feature Engineering Pipeline for MPEG-G Microbiome Challenge
========================================================================

This script implements comprehensive feature engineering to improve upon the baseline 30% accuracy
by addressing overfitting and small sample size issues through biologically meaningful features.

Key Features:
1. Temporal Features (T1/T2 analysis)
2. Diversity Metrics (Alpha/Beta diversity)
3. Functional Aggregation
4. Interaction Features
5. Dimensionality Reduction
6. Advanced Statistical Transformations

Author: Sayantan Pal
Date: 2025-09-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering pipeline for microbiome data analysis.
    
    Focuses on creating biologically meaningful features that work with small sample sizes
    while addressing overfitting through dimensionality reduction and robust transformations.
    """
    
    def __init__(self, output_dir: str = "enhanced_features", verbose: bool = True):
        self.output_dir = output_dir
        self.verbose = verbose
        self.feature_importance_ = {}
        self.biological_interpretation_ = {}
        
        # Setup logging
        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/feature_engineering.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, features_path: str, metadata_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate microbiome data."""
        self.logger.info("Loading microbiome data...")
        
        # Load features and metadata
        features = pd.read_csv(features_path, index_col=0)
        metadata = pd.read_csv(metadata_path, index_col=0)
        
        # Ensure alignment
        common_samples = features.index.intersection(metadata.index)
        features = features.loc[common_samples]
        metadata = metadata.loc[common_samples]
        
        self.logger.info(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
        self.logger.info(f"Class distribution: {metadata['symptom'].value_counts().to_dict()}")
        
        return features, metadata
    
    def extract_temporal_features(self, features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from T1/T2 timepoints.
        
        Creates features for:
        - Change scores (T2 - T1)
        - Relative change ratios (T2/T1) 
        - Stability metrics
        - Timepoint interactions
        """
        self.logger.info("Extracting temporal features...")
        
        temporal_features = []
        feature_names = []
        
        # Group by subject to get paired timepoints
        subjects = metadata.groupby('subject_id')
        
        for subject_id, subject_data in subjects:
            if len(subject_data) == 2:  # Ensure we have both T1 and T2
                t1_idx = subject_data[subject_data['timepoint'] == 'T1'].index[0]
                t2_idx = subject_data[subject_data['timepoint'] == 'T2'].index[0]
                
                t1_features = features.loc[t1_idx]
                t2_features = features.loc[t2_idx]
                
                # 1. Change scores (T2 - T1)
                change_scores = t2_features - t1_features
                
                # 2. Relative change ratios (T2/T1) with pseudocount
                pseudocount = 1e-6
                relative_changes = (t2_features + pseudocount) / (t1_features + pseudocount)
                
                # 3. Stability metrics (absolute change)
                stability = np.abs(change_scores)
                
                # 4. Temporal variance
                temporal_var = np.var([t1_features, t2_features], axis=0)
                
                # Combine features for this subject
                subject_temporal = np.concatenate([
                    change_scores.values,
                    np.log(relative_changes.values),  # Log-transform ratios
                    stability.values,
                    temporal_var
                ])
                
                temporal_features.append(subject_temporal)
                
                # Store feature names (do this once)
                if len(feature_names) == 0:
                    original_names = features.columns
                    feature_names = (
                        [f'change_{name}' for name in original_names] +
                        [f'log_ratio_{name}' for name in original_names] +
                        [f'stability_{name}' for name in original_names] +
                        [f'temporal_var_{name}' for name in original_names]
                    )
        
        # Create DataFrame with subject-level temporal features
        temporal_df = pd.DataFrame(temporal_features, columns=feature_names)
        
        # Add subject metadata (use T1 as reference)
        subject_metadata = []
        for subject_id, subject_data in subjects:
            if len(subject_data) == 2:
                t1_meta = subject_data[subject_data['timepoint'] == 'T1'].iloc[0]
                subject_metadata.append(t1_meta)
        
        subject_meta_df = pd.DataFrame(subject_metadata)
        temporal_df.index = subject_meta_df.index
        
        self.logger.info(f"Created {temporal_df.shape[1]} temporal features for {temporal_df.shape[0]} subjects")
        
        return temporal_df, subject_meta_df
    
    def calculate_diversity_metrics(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate alpha diversity metrics.
        
        Metrics include:
        - Shannon diversity
        - Simpson diversity  
        - Observed species count
        - Pielou's evenness
        - Chao1 richness estimator
        """
        self.logger.info("Calculating diversity metrics...")
        
        diversity_metrics = []
        
        for idx, sample in features.iterrows():
            # Convert to relative abundance (sum to 1)
            # Add pseudocount to handle log-transformed negative values
            rel_abundance = np.exp(sample - sample.max())  # Shift to positive
            rel_abundance = rel_abundance / rel_abundance.sum()
            
            # Remove zeros for calculations
            nonzero_abundance = rel_abundance[rel_abundance > 0]
            
            # Shannon diversity
            shannon = entropy(nonzero_abundance, base=2)
            
            # Simpson diversity (1 - Simpson index)
            simpson = 1 - np.sum(nonzero_abundance ** 2)
            
            # Observed species (richness)
            observed = len(nonzero_abundance)
            
            # Pielou's evenness
            max_shannon = np.log2(observed) if observed > 1 else 1
            pielou = shannon / max_shannon if max_shannon > 0 else 0
            
            # Chao1 richness estimator (simplified)
            singletons = np.sum(rel_abundance <= 1/len(rel_abundance))
            doubletons = np.sum((rel_abundance > 1/len(rel_abundance)) & 
                              (rel_abundance <= 2/len(rel_abundance)))
            if doubletons > 0:
                chao1 = observed + (singletons ** 2) / (2 * doubletons)
            else:
                chao1 = observed
            
            diversity_metrics.append([shannon, simpson, observed, pielou, chao1])
        
        diversity_df = pd.DataFrame(
            diversity_metrics,
            columns=['shannon_diversity', 'simpson_diversity', 'observed_species', 
                    'pielou_evenness', 'chao1_richness'],
            index=features.index
        )
        
        self.logger.info(f"Calculated {diversity_df.shape[1]} diversity metrics")
        
        return diversity_df
    
    def create_functional_aggregations(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Create functional aggregation features.
        
        Aggregates related functions and creates ratios between functional categories.
        """
        self.logger.info("Creating functional aggregations...")
        
        # Separate species and function features
        species_cols = [col for col in features.columns if col.startswith('species_')]
        function_cols = [col for col in features.columns if col.startswith('function_')]
        
        aggregated_features = []
        feature_names = []
        
        for idx, sample in features.iterrows():
            sample_aggregations = []
            
            # Species aggregations
            species_data = sample[species_cols]
            
            # Total species abundance
            total_species = np.sum(np.exp(species_data))  # Convert from log
            
            # Top species abundance (top 10%)
            top_species_count = max(1, len(species_data) // 10)
            top_species = np.sum(np.exp(species_data.nlargest(top_species_count)))
            
            # Species concentration ratio (top 10% / total)
            species_concentration = top_species / total_species if total_species > 0 else 0
            
            sample_aggregations.extend([total_species, top_species, species_concentration])
            
            # Function aggregations
            function_data = sample[function_cols]
            
            # Group functions by pathway (simplified by K number ranges)
            metabolism_functions = function_data[[col for col in function_data.index 
                                                if any(k in col for k in ['K00', 'K01', 'K02'])]]
            information_functions = function_data[[col for col in function_data.index 
                                                 if any(k in col for k in ['K03', 'K04'])]]
            transport_functions = function_data[[col for col in function_data.index 
                                              if any(k in col for k in ['K05', 'K06'])]]
            
            # Pathway abundances
            metabolism_abundance = np.sum(np.exp(metabolism_functions)) if len(metabolism_functions) > 0 else 0
            information_abundance = np.sum(np.exp(information_functions)) if len(information_functions) > 0 else 0
            transport_abundance = np.sum(np.exp(transport_functions)) if len(transport_functions) > 0 else 0
            
            total_function = metabolism_abundance + information_abundance + transport_abundance
            
            # Pathway ratios
            if total_function > 0:
                metabolism_ratio = metabolism_abundance / total_function
                information_ratio = information_abundance / total_function
                transport_ratio = transport_abundance / total_function
            else:
                metabolism_ratio = information_ratio = transport_ratio = 0
            
            sample_aggregations.extend([
                metabolism_abundance, information_abundance, transport_abundance,
                metabolism_ratio, information_ratio, transport_ratio
            ])
            
            aggregated_features.append(sample_aggregations)
            
            # Feature names (do this once)
            if len(feature_names) == 0:
                feature_names = [
                    'total_species_abundance', 'top_species_abundance', 'species_concentration_ratio',
                    'metabolism_abundance', 'information_abundance', 'transport_abundance',
                    'metabolism_ratio', 'information_ratio', 'transport_ratio'
                ]
        
        aggregated_df = pd.DataFrame(aggregated_features, columns=feature_names, index=features.index)
        
        self.logger.info(f"Created {aggregated_df.shape[1]} functional aggregation features")
        
        return aggregated_df
    
    def create_interaction_features(self, features: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
        """
        Create interaction features between top species and functions.
        
        Creates ratios and products between the most important features.
        """
        self.logger.info("Creating interaction features...")
        
        # Select top features based on variance (more stable than statistical tests with small samples)
        feature_vars = features.var().sort_values(ascending=False)
        top_features = feature_vars.head(top_k).index.tolist()
        
        interaction_features = []
        feature_names = []
        
        for idx, sample in features.iterrows():
            sample_interactions = []
            
            # Create pairwise ratios for top features
            for i in range(min(10, len(top_features))):  # Limit to prevent explosion
                for j in range(i+1, min(10, len(top_features))):
                    feat1, feat2 = top_features[i], top_features[j]
                    
                    # Ratio (with pseudocount)
                    pseudocount = 1e-6
                    ratio = (sample[feat1] + pseudocount) / (sample[feat2] + pseudocount)
                    
                    # Product
                    product = sample[feat1] * sample[feat2]
                    
                    sample_interactions.extend([ratio, product])
                    
                    # Feature names (do this once)
                    if len(feature_names) < len(sample_interactions):
                        feature_names.extend([
                            f'ratio_{feat1}_vs_{feat2}',
                            f'product_{feat1}_and_{feat2}'
                        ])
            
            interaction_features.append(sample_interactions)
        
        interaction_df = pd.DataFrame(interaction_features, columns=feature_names, index=features.index)
        
        self.logger.info(f"Created {interaction_df.shape[1]} interaction features")
        
        return interaction_df
    
    def apply_dimensionality_reduction(self, features: pd.DataFrame, n_components: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Apply multiple dimensionality reduction techniques.
        
        Returns PCA, t-SNE, and UMAP embeddings.
        """
        self.logger.info("Applying dimensionality reduction...")
        
        # Standardize features for dimensionality reduction
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        reduction_results = {}
        
        # PCA
        pca = PCA(n_components=min(n_components, features.shape[0]-1, features.shape[1]))
        pca_features = pca.fit_transform(features_scaled)
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'pca_component_{i+1}' for i in range(pca_features.shape[1])],
            index=features.index
        )
        reduction_results['pca'] = pca_df
        
        # Store PCA explained variance for interpretation
        self.feature_importance_['pca_explained_variance'] = pca.explained_variance_ratio_
        
        # t-SNE (for visualization, fewer components)
        tsne_components = min(3, n_components)
        tsne = TSNE(n_components=tsne_components, random_state=42, perplexity=min(30, features.shape[0]//4))
        tsne_features = tsne.fit_transform(features_scaled)
        tsne_df = pd.DataFrame(
            tsne_features,
            columns=[f'tsne_component_{i+1}' for i in range(tsne_features.shape[1])],
            index=features.index
        )
        reduction_results['tsne'] = tsne_df
        
        # UMAP
        umap_components = min(10, n_components)
        umap_reducer = umap.UMAP(n_components=umap_components, random_state=42, 
                                n_neighbors=min(15, features.shape[0]//3))
        umap_features = umap_reducer.fit_transform(features_scaled)
        umap_df = pd.DataFrame(
            umap_features,
            columns=[f'umap_component_{i+1}' for i in range(umap_features.shape[1])],
            index=features.index
        )
        reduction_results['umap'] = umap_df
        
        self.logger.info(f"Applied dimensionality reduction: PCA ({pca_df.shape[1]}), "
                        f"t-SNE ({tsne_df.shape[1]}), UMAP ({umap_df.shape[1]})")
        
        return reduction_results
    
    def apply_statistical_transformations(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply advanced statistical transformations.
        
        Includes CLR transformation, robust scaling, and outlier-resistant features.
        """
        self.logger.info("Applying statistical transformations...")
        
        transformed_features = []
        feature_names = []
        
        for idx, sample in features.iterrows():
            sample_transforms = []
            
            # Convert from log scale back to counts (approximately)
            counts = np.exp(sample - sample.min())  # Shift to positive
            
            # Centered Log-Ratio (CLR) transformation
            geometric_mean = stats.gmean(counts + 1)  # Add pseudocount
            clr_values = np.log(counts + 1) - np.log(geometric_mean)
            
            # Robust z-scores (using median and MAD)
            median_val = np.median(sample)
            mad_val = stats.median_abs_deviation(sample)
            robust_z = (sample - median_val) / (mad_val + 1e-6)
            
            # Quantile features
            q25, q50, q75 = np.percentile(sample, [25, 50, 75])
            iqr = q75 - q25
            
            # Statistical moments
            skewness = stats.skew(sample)
            kurtosis = stats.kurtosis(sample)
            
            # Outlier counts (using IQR method)
            outlier_threshold = 1.5 * iqr
            outliers_low = np.sum(sample < (q25 - outlier_threshold))
            outliers_high = np.sum(sample > (q75 + outlier_threshold))
            
            sample_transforms.extend([
                np.mean(clr_values), np.std(clr_values),  # CLR summary
                np.mean(robust_z), np.std(robust_z),     # Robust scaling summary
                q25, q50, q75, iqr,                      # Quantiles
                skewness, kurtosis,                      # Moments
                outliers_low, outliers_high              # Outlier counts
            ])
            
            transformed_features.append(sample_transforms)
            
            # Feature names (do this once)
            if len(feature_names) == 0:
                feature_names = [
                    'clr_mean', 'clr_std', 'robust_z_mean', 'robust_z_std',
                    'q25', 'q50', 'q75', 'iqr', 'skewness', 'kurtosis',
                    'outliers_low', 'outliers_high'
                ]
        
        transformed_df = pd.DataFrame(transformed_features, columns=feature_names, index=features.index)
        
        self.logger.info(f"Created {transformed_df.shape[1]} statistical transformation features")
        
        return transformed_df
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, method: str = 'mutual_info', 
                       k: int = 100) -> pd.DataFrame:
        """
        Select the most informative features using various selection methods.
        """
        self.logger.info(f"Selecting top {k} features using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, features.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(k, features.shape[1]))
        elif method == 'random_forest':
            # Use Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, target)
            
            # Get top k features
            importances = pd.Series(rf.feature_importances_, index=features.columns)
            top_features = importances.nlargest(min(k, len(importances))).index
            
            self.feature_importance_['random_forest'] = importances.sort_values(ascending=False)
            
            return features[top_features]
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        selected_features = selector.fit_transform(features, target)
        selected_columns = features.columns[selector.get_support()]
        
        # Store feature scores
        if hasattr(selector, 'scores_'):
            scores = pd.Series(selector.scores_, index=features.columns)
            self.feature_importance_[method] = scores.sort_values(ascending=False)
        
        selected_df = pd.DataFrame(selected_features, columns=selected_columns, index=features.index)
        
        self.logger.info(f"Selected {selected_df.shape[1]} features")
        
        return selected_df
    
    def create_biological_interpretations(self, feature_importance: Dict[str, pd.Series]) -> Dict[str, str]:
        """
        Create biological interpretations for top features.
        """
        interpretations = {}
        
        # KEGG function interpretations (simplified)
        kegg_descriptions = {
            'K00105': 'Malate dehydrogenase - Central metabolism',
            'K00298': 'Propionyl-CoA carboxylase - Fatty acid metabolism',
            'K00311': 'Electron transfer - Energy metabolism',
            'K00529': 'Ribose metabolism - Sugar processing',
            'K00683': 'Methionine synthesis - Amino acid metabolism',
            'K02245': 'ABC transporter - Nutrient uptake',
            'K03194': 'DNA repair - Genome maintenance',
            'K03309': 'DNA polymerase - Replication',
            'K03379': 'Cell division - Growth control',
            'K03746': 'Ribosomal protein - Protein synthesis'
        }
        
        for method, importance_scores in feature_importance.items():
            # Handle different types of importance scores
            if isinstance(importance_scores, pd.Series):
                top_features = importance_scores.head(10)
            elif isinstance(importance_scores, np.ndarray):
                # Skip arrays that don't have feature names
                continue
            else:
                continue
            method_interpretations = []
            
            for feature, score in top_features.items():
                if 'function_' in feature:
                    k_num = feature.replace('function_', '')
                    description = kegg_descriptions.get(k_num, 'Unknown function')
                    method_interpretations.append(f"{feature}: {description} (importance: {score:.3f})")
                elif 'species_' in feature:
                    species = feature.replace('species_', '').replace('_', ' ')
                    method_interpretations.append(f"{feature}: {species} abundance (importance: {score:.3f})")
                elif any(keyword in feature for keyword in ['diversity', 'shannon', 'simpson']):
                    method_interpretations.append(f"{feature}: Microbial diversity metric (importance: {score:.3f})")
                elif 'temporal' in feature or 'change' in feature:
                    method_interpretations.append(f"{feature}: Temporal stability/change (importance: {score:.3f})")
                else:
                    method_interpretations.append(f"{feature}: Engineered feature (importance: {score:.3f})")
            
            interpretations[method] = '\n'.join(method_interpretations)
        
        return interpretations
    
    def generate_visualizations(self, enhanced_features: pd.DataFrame, metadata: pd.DataFrame,
                              reduction_results: Dict[str, pd.DataFrame]):
        """
        Generate comprehensive visualizations for enhanced features.
        """
        self.logger.info("Generating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. PCA visualization
        pca_data = reduction_results['pca']
        colors = {'Healthy': 'green', 'Mild': 'yellow', 'Moderate': 'orange', 'Severe': 'red'}
        for symptom in colors.keys():
            mask = metadata['symptom'] == symptom
            if mask.sum() > 0:
                axes[0,0].scatter(pca_data.loc[mask, 'pca_component_1'], 
                                pca_data.loc[mask, 'pca_component_2'],
                                c=colors[symptom], label=symptom, alpha=0.7)
        axes[0,0].set_xlabel('PC1')
        axes[0,0].set_ylabel('PC2')
        axes[0,0].set_title('PCA Visualization')
        axes[0,0].legend()
        
        # 2. Feature importance heatmap
        if 'random_forest' in self.feature_importance_:
            top_features = self.feature_importance_['random_forest'].head(20)
            axes[0,1].barh(range(len(top_features)), top_features.values)
            axes[0,1].set_yticks(range(len(top_features)))
            axes[0,1].set_yticklabels([f.replace('function_', '').replace('species_', '')[:15] 
                                     for f in top_features.index])
            axes[0,1].set_title('Top 20 Feature Importance')
            axes[0,1].set_xlabel('Importance Score')
        
        # 3. Class distribution
        class_counts = metadata['symptom'].value_counts()
        axes[0,2].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[0,2].set_title('Class Distribution')
        
        # 4. Feature correlation heatmap (top features)
        if enhanced_features.shape[1] > 10:
            corr_features = enhanced_features.iloc[:, :20]  # Top 20 features
            corr_matrix = corr_features.corr()
            sns.heatmap(corr_matrix, ax=axes[1,0], cmap='coolwarm', center=0, square=True)
            axes[1,0].set_title('Feature Correlation Matrix')
        
        # 5. Diversity metrics by class
        diversity_cols = [col for col in enhanced_features.columns if 'diversity' in col]
        if diversity_cols:
            diversity_data = enhanced_features[diversity_cols[0]]  # Shannon diversity
            symptom_diversity = []
            for symptom in colors.keys():
                mask = metadata['symptom'] == symptom
                if mask.sum() > 0:
                    symptom_diversity.append(diversity_data[mask].values)
            
            axes[1,1].boxplot(symptom_diversity, labels=colors.keys())
            axes[1,1].set_title('Diversity by Symptom Severity')
            axes[1,1].set_ylabel('Shannon Diversity')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Temporal features (if available)
        temporal_cols = [col for col in enhanced_features.columns if 'change' in col]
        if temporal_cols and len(temporal_cols) > 0:
            temporal_data = enhanced_features[temporal_cols[0]]
            axes[1,2].hist(temporal_data.dropna(), bins=20, alpha=0.7)
            axes[1,2].set_title('Distribution of Temporal Changes')
            axes[1,2].set_xlabel('Change Score')
            axes[1,2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/enhanced_features_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PCA explained variance plot
        if 'pca_explained_variance' in self.feature_importance_:
            plt.figure(figsize=(10, 6))
            variance_ratio = self.feature_importance_['pca_explained_variance']
            cumulative_variance = np.cumsum(variance_ratio)
            
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
            plt.axhline(y=0.8, color='r', linestyle='--', label='80% variance')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/pca_explained_variance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_pipeline(self, features_path: str, metadata_path: str, 
                    use_temporal: bool = True, final_k: int = 200) -> Dict[str, pd.DataFrame]:
        """
        Run the complete advanced feature engineering pipeline.
        
        Returns enhanced feature matrices ready for ML training.
        """
        self.logger.info("Starting advanced feature engineering pipeline...")
        
        # Load data
        features, metadata = self.load_data(features_path, metadata_path)
        
        # Initialize results
        all_features = []
        feature_sets = {}
        
        # 1. Original features (subset for baseline)
        original_subset = self.select_features(features, metadata['symptom'], 
                                             method='random_forest', k=100)
        all_features.append(original_subset)
        feature_sets['original_subset'] = original_subset
        
        # 2. Diversity metrics
        diversity_features = self.calculate_diversity_metrics(features)
        all_features.append(diversity_features)
        feature_sets['diversity'] = diversity_features
        
        # 3. Functional aggregations
        functional_features = self.create_functional_aggregations(features)
        all_features.append(functional_features)
        feature_sets['functional'] = functional_features
        
        # 4. Interaction features
        interaction_features = self.create_interaction_features(features, top_k=30)
        all_features.append(interaction_features)
        feature_sets['interactions'] = interaction_features
        
        # 5. Statistical transformations
        statistical_features = self.apply_statistical_transformations(features)
        all_features.append(statistical_features)
        feature_sets['statistical'] = statistical_features
        
        # 6. Temporal features (if requested and data supports it)
        if use_temporal and 'timepoint' in metadata.columns:
            try:
                temporal_features, temporal_metadata = self.extract_temporal_features(features, metadata)
                # For temporal features, we need to update metadata to subject-level
                metadata_for_temporal = temporal_metadata
                target_for_temporal = temporal_metadata['symptom']
                
                # Apply other feature engineering to temporal data
                temporal_diversity = self.calculate_diversity_metrics(temporal_features)
                temporal_statistical = self.apply_statistical_transformations(temporal_features)
                
                temporal_combined = pd.concat([temporal_features, temporal_diversity, 
                                             temporal_statistical], axis=1)
                
                # Select best temporal features
                temporal_selected = self.select_features(temporal_combined, target_for_temporal,
                                                       method='random_forest', k=50)
                
                feature_sets['temporal'] = temporal_selected
                
                # Save temporal-specific outputs
                temporal_selected.to_csv(f'{self.output_dir}/temporal_enhanced_features.csv')
                temporal_metadata.to_csv(f'{self.output_dir}/temporal_metadata.csv')
                
                self.logger.info(f"Created temporal features: {temporal_selected.shape}")
                
            except Exception as e:
                self.logger.warning(f"Could not create temporal features: {e}")
                use_temporal = False
        
        # Combine all features
        if use_temporal and 'temporal' in feature_sets:
            # For temporal analysis, use temporal features
            combined_features = feature_sets['temporal']
            final_metadata = temporal_metadata
        else:
            # For sample-level analysis, combine all non-temporal features
            combined_features = pd.concat(all_features, axis=1)
            final_metadata = metadata
        
        # 7. Dimensionality reduction
        reduction_results = self.apply_dimensionality_reduction(combined_features, n_components=20)
        
        # Add dimensionality reduction features
        for method, reduced_features in reduction_results.items():
            if method == 'pca':  # Focus on PCA for final features
                combined_features = pd.concat([combined_features, reduced_features], axis=1)
        
        # 8. Final feature selection
        final_features = self.select_features(combined_features, final_metadata['symptom'],
                                            method='random_forest', k=final_k)
        
        # 9. Create biological interpretations
        self.biological_interpretation_ = self.create_biological_interpretations(self.feature_importance_)
        
        # 10. Generate visualizations
        self.generate_visualizations(final_features, final_metadata, reduction_results)
        
        # Save results
        final_features.to_csv(f'{self.output_dir}/enhanced_features_final.csv')
        final_metadata.to_csv(f'{self.output_dir}/enhanced_metadata_final.csv')
        
        # Save feature importance and interpretations
        with open(f'{self.output_dir}/feature_importance.json', 'w') as f:
            # Convert pandas Series to dict for JSON serialization
            importance_dict = {}
            for method, series in self.feature_importance_.items():
                if isinstance(series, pd.Series):
                    importance_dict[method] = series.to_dict()
                elif isinstance(series, np.ndarray):
                    importance_dict[method] = series.tolist()
                else:
                    # Convert other types to string for safety
                    importance_dict[method] = str(series)
            json.dump(importance_dict, f, indent=2)
        
        with open(f'{self.output_dir}/biological_interpretations.json', 'w') as f:
            json.dump(self.biological_interpretation_, f, indent=2)
        
        # Create summary report
        self.create_summary_report(final_features, final_metadata, feature_sets)
        
        self.logger.info(f"Pipeline completed! Enhanced features shape: {final_features.shape}")
        
        return {
            'enhanced_features': final_features,
            'metadata': final_metadata,
            'feature_sets': feature_sets,
            'reduction_results': reduction_results
        }
    
    def create_summary_report(self, final_features: pd.DataFrame, metadata: pd.DataFrame,
                            feature_sets: Dict[str, pd.DataFrame]):
        """
        Create a comprehensive summary report.
        """
        report = []
        report.append("ADVANCED FEATURE ENGINEERING SUMMARY REPORT")
        report.append("=" * 50)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data overview
        report.append("DATA OVERVIEW:")
        report.append(f"- Total samples: {final_features.shape[0]}")
        report.append(f"- Final features: {final_features.shape[1]}")
        report.append(f"- Class distribution: {metadata['symptom'].value_counts().to_dict()}")
        report.append("")
        
        # Feature set breakdown
        report.append("FEATURE SET BREAKDOWN:")
        for name, df in feature_sets.items():
            report.append(f"- {name}: {df.shape[1]} features")
        report.append("")
        
        # Top features by importance
        if 'random_forest' in self.feature_importance_:
            report.append("TOP 20 MOST IMPORTANT FEATURES:")
            top_features = self.feature_importance_['random_forest'].head(20)
            for i, (feature, importance) in enumerate(top_features.items(), 1):
                report.append(f"{i:2d}. {feature}: {importance:.4f}")
            report.append("")
        
        # Biological interpretations
        report.append("BIOLOGICAL INTERPRETATIONS:")
        for method, interpretation in self.biological_interpretation_.items():
            report.append(f"\n{method.upper()}:")
            report.append(interpretation)
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS FOR MODEL IMPROVEMENT:")
        report.append("1. Use ensemble methods to combine multiple feature perspectives")
        report.append("2. Apply advanced regularization (L1/L2/Elastic Net) to prevent overfitting")
        report.append("3. Consider stratified cross-validation due to class imbalance")
        report.append("4. Explore semi-supervised learning with unlabeled samples")
        report.append("5. Use feature importance for biological hypothesis generation")
        report.append("6. Consider temporal modeling if longitudinal data is available")
        report.append("7. Apply SMOTE or other balancing techniques for minority classes")
        report.append("")
        
        # Technical notes
        report.append("TECHNICAL NOTES:")
        report.append("- Features are already scaled and normalized")
        report.append("- Dimensionality reduction helps with small sample size")
        report.append("- Interaction features capture non-linear relationships")
        report.append("- Diversity metrics provide ecological context")
        report.append("- Statistical transformations improve robustness")
        
        # Save report
        with open(f'{self.output_dir}/ENHANCEMENT_SUMMARY.txt', 'w') as f:
            f.write('\n'.join(report))
        
        self.logger.info("Summary report created!")


def main():
    """Main execution function."""
    
    # Setup paths
    base_dir = "/Users/sayantanpal100/Desktop/MPEG-G_ Decoding the Dialogue"
    features_path = f"{base_dir}/processed_data/microbiome_features_processed.csv"
    metadata_path = f"{base_dir}/processed_data/microbiome_metadata_processed.csv"
    output_dir = f"{base_dir}/enhanced_features"
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer(output_dir=output_dir, verbose=True)
    
    # Run pipeline
    results = engineer.run_pipeline(
        features_path=features_path,
        metadata_path=metadata_path,
        use_temporal=True,
        final_k=150  # Reasonable number for 40 samples
    )
    
    print("\nAdvanced Feature Engineering Complete!")
    print(f"Enhanced features shape: {results['enhanced_features'].shape}")
    print(f"Results saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Use enhanced_features_final.csv for ML training")
    print("2. Review feature_importance.json for model insights")
    print("3. Check biological_interpretations.json for scientific understanding")
    print("4. Examine ENHANCEMENT_SUMMARY.txt for recommendations")


if __name__ == "__main__":
    main()