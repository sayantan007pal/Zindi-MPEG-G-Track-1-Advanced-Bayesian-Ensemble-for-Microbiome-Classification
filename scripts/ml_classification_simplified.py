#!/usr/bin/env python3
"""
Simplified Classification ML Pipeline for MPEG-G Challenge Track 1
Optimized for small datasets with high-dimensional features
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict

# ML Libraries
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix
)
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedMicrobiomeClassifier:
    """
    Simplified ML pipeline optimized for small microbiome datasets
    """
    
    def __init__(self, 
                 feature_matrix_path: str,
                 metadata_path: str,
                 output_dir: str = './model_outputs_microbiome',
                 random_seed: int = 42,
                 n_features: int = 100):
        
        self.feature_matrix_path = feature_matrix_path
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.n_features = n_features
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Set random seeds
        np.random.seed(random_seed)
        
        # Set up logging
        log_file = self.output_dir / f'simplified_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and prepare data"""
        logger.info("Loading data...")
        
        # Load feature matrix
        self.feature_matrix = pd.read_csv(self.feature_matrix_path, index_col=0)
        logger.info(f"Loaded feature matrix: {self.feature_matrix.shape}")
        
        # Load metadata with targets
        self.metadata = pd.read_csv(self.metadata_path, index_col=0)
        logger.info(f"Loaded metadata: {self.metadata.shape}")
        
        # Align indices
        common_indices = self.feature_matrix.index.intersection(self.metadata.index)
        self.feature_matrix = self.feature_matrix.loc[common_indices]
        self.metadata = self.metadata.loc[common_indices]
        
        logger.info(f"Aligned samples: {len(common_indices)}")
        
        # Get microbiome features
        self.species_features = [col for col in self.feature_matrix.columns 
                               if col.startswith('species_')]
        self.function_features = [col for col in self.feature_matrix.columns 
                                if col.startswith('function_')]
        
        logger.info(f"Species features: {len(self.species_features)}")
        logger.info(f"Function features: {len(self.function_features)}")
        
        # Encode target labels
        self.label_encoder = LabelEncoder()
        self.metadata['symptom_encoded'] = self.label_encoder.fit_transform(self.metadata['symptom'])
        
        # Get class distribution
        class_dist = self.metadata['symptom'].value_counts()
        logger.info(f"Class distribution: {class_dist.to_dict()}")
        
        self.n_classes = len(self.label_encoder.classes_)
        self.class_names = self.label_encoder.classes_
        logger.info(f"Number of classes: {self.n_classes}")
        logger.info(f"Class names: {list(self.class_names)}")
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified feature preprocessing
        """
        logger.info("Preprocessing features...")
        
        # Fill NaN values
        X_clean = X.fillna(0)
        
        # Log transform species abundances (add 1 to avoid log(0))
        species_cols = [col for col in X_clean.columns if col.startswith('species_')]
        X_clean[species_cols] = np.log1p(X_clean[species_cols].clip(lower=0))
        
        # Create simple diversity metrics
        diversity_features = {}
        
        if species_cols:
            # Simple richness (number of non-zero species)
            diversity_features['species_richness'] = (X_clean[species_cols] > 0).sum(axis=1).values
            
            # Total abundance
            diversity_features['total_abundance'] = X_clean[species_cols].sum(axis=1).values
            
            # Shannon-like diversity (simplified)
            abundance_data = X_clean[species_cols].values + 1e-10
            proportions = abundance_data / abundance_data.sum(axis=1, keepdims=True)
            shannon = -np.sum(proportions * np.log(proportions), axis=1)
            diversity_features['shannon_diversity'] = shannon
        
        # Add diversity features
        if diversity_features:
            diversity_df = pd.DataFrame(diversity_features, index=X_clean.index)
            X_clean = pd.concat([X_clean, diversity_df], axis=1)
        
        # Remove features with zero variance
        variance_threshold = 1e-6
        feature_vars = X_clean.var()
        low_var_features = feature_vars[feature_vars < variance_threshold].index
        X_clean = X_clean.drop(columns=low_var_features)
        
        logger.info(f"Removed {len(low_var_features)} low-variance features")
        logger.info(f"Final feature matrix shape: {X_clean.shape}")
        
        return X_clean
    
    def select_features(self, X_train, y_train, X_test):
        """
        Select top features using statistical tests
        """
        logger.info(f"Selecting top {self.n_features} features...")
        
        # Use different selectors for different feature types
        selector = SelectKBest(score_func=f_classif, k=min(self.n_features, X_train.shape[1]))
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_features = X_train.columns[selector.get_support()].tolist()
        logger.info(f"Selected {len(selected_features)} features")
        
        return X_train_selected, X_test_selected, selected_features, selector
    
    def train_models(self, X_train, y_train, X_test, y_test, feature_names):
        """
        Train simplified set of models
        """
        results = {}
        
        # Set up cross-validation (reduce folds for small dataset)
        cv_folds = min(3, len(np.unique(y_train)))  # Ensure each fold has all classes
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        
        # 1. Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=5,      # Prevent overfitting on small dataset
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=self.random_seed,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)
        
        results['random_forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'probabilities': rf_proba,
            'accuracy': accuracy_score(y_test, rf_pred),
            'cv_accuracy': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'feature_importance': pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        # 2. LightGBM (usually works well with small datasets)
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,
            num_leaves=15,    # Reduced to prevent overfitting
            learning_rate=0.1,
            min_child_samples=5,  # Prevent overfitting
            random_state=self.random_seed,
            class_weight='balanced',
            verbosity=-1
        )
        
        cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)
        
        results['lightgbm'] = {
            'model': lgb_model,
            'predictions': lgb_pred,
            'probabilities': lgb_proba,
            'accuracy': accuracy_score(y_test, lgb_pred),
            'cv_accuracy': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'feature_importance': pd.DataFrame({
                'feature': feature_names,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        # 3. Logistic Regression (good baseline for small datasets)
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=self.random_seed,
            max_iter=1000,
            class_weight='balanced',
            C=0.1  # Regularization to prevent overfitting
        )
        
        cv_scores = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_proba = lr_model.predict_proba(X_test)
        
        results['logistic_regression'] = {
            'model': lr_model,
            'predictions': lr_pred,
            'probabilities': lr_proba,
            'accuracy': accuracy_score(y_test, lr_pred),
            'cv_accuracy': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std()
        }
        
        return results
    
    def ensemble_predictions(self, models_dict: Dict):
        """
        Simple ensemble using accuracy-weighted averaging
        """
        logger.info("Creating ensemble predictions...")
        
        # Collect predictions and weights
        all_probabilities = []
        weights = []
        
        for model_name, model_info in models_dict.items():
            all_probabilities.append(model_info['probabilities'])
            weights.append(model_info['cv_accuracy'])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average of probabilities
        ensemble_proba = np.average(all_probabilities, axis=0, weights=weights)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, ensemble_proba
    
    def create_visualizations(self, results: Dict, y_test):
        """
        Create comprehensive visualizations
        """
        logger.info("Creating visualizations...")
        
        # Model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.keys())
        if 'ensemble' in results:
            models.remove('ensemble')
        
        # Test accuracies
        test_accuracies = [results[m]['accuracy'] for m in models]
        cv_accuracies = [results[m]['cv_accuracy'] for m in models]
        
        axes[0, 0].bar(models, test_accuracies, color='skyblue', alpha=0.7, label='Test Accuracy')
        axes[0, 0].bar(models, cv_accuracies, color='orange', alpha=0.7, label='CV Accuracy')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confusion matrix for best model
        best_model = max(models, key=lambda x: results[x]['accuracy'])
        best_pred = results[best_model]['predictions']
        
        cm = confusion_matrix(y_test, best_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # Feature importance for tree-based models
        if 'random_forest' in results:
            importance_df = results['random_forest']['feature_importance'].head(15)
            axes[1, 0].barh(range(len(importance_df)), importance_df['importance'])
            axes[1, 0].set_yticks(range(len(importance_df)))
            axes[1, 0].set_yticklabels(importance_df['feature'], fontsize=8)
            axes[1, 0].set_title('Top 15 Feature Importances (Random Forest)')
            axes[1, 0].set_xlabel('Importance')
        
        # Class distribution
        class_counts = self.metadata['symptom'].value_counts()
        axes[1, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Class Distribution in Dataset')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations/comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual confusion matrices
        for model_name in models + (['ensemble'] if 'ensemble' in results else []):
            if model_name in results:
                pred = results[model_name]['predictions']
                cm = confusion_matrix(y_test, pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=self.class_names,
                           yticklabels=self.class_names)
                plt.title(f'Confusion Matrix - {model_name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'visualizations/confusion_matrix_{model_name}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self, results: Dict, y_test, selected_features):
        """
        Generate comprehensive classification report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': len(self.feature_matrix),
                'n_original_features': self.feature_matrix.shape[1],
                'n_selected_features': len(selected_features),
                'n_classes': self.n_classes,
                'class_names': list(self.class_names),
                'class_distribution': self.metadata['symptom'].value_counts().to_dict()
            },
            'model_performance': {},
            'best_model': None,
            'best_accuracy': 0,
            'selected_features': selected_features[:20]  # Top 20 features
        }
        
        for model_name, model_info in results.items():
            if model_name != 'ensemble':
                y_pred = model_info['predictions']
                
                # Calculate detailed metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='macro', zero_division=0
                )
                
                class_report = classification_report(
                    y_test, y_pred, 
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0
                )
                
                report['model_performance'][model_name] = {
                    'accuracy': float(model_info['accuracy']),
                    'cv_accuracy': float(model_info['cv_accuracy']),
                    'cv_accuracy_std': float(model_info['cv_accuracy_std']),
                    'precision_macro': float(precision),
                    'recall_macro': float(recall),
                    'f1_macro': float(f1),
                    'per_class_metrics': class_report
                }
                
                if model_info['accuracy'] > report['best_accuracy']:
                    report['best_model'] = model_name
                    report['best_accuracy'] = float(model_info['accuracy'])
        
        # Handle ensemble if present
        if 'ensemble' in results:
            ensemble_info = results['ensemble']
            y_pred = ensemble_info['predictions']
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )
            
            class_report = classification_report(
                y_test, y_pred, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            report['model_performance']['ensemble'] = {
                'accuracy': float(ensemble_info['accuracy']),
                'precision_macro': float(precision),
                'recall_macro': float(recall),
                'f1_macro': float(f1),
                'per_class_metrics': class_report
            }
            
            if ensemble_info['accuracy'] > report['best_accuracy']:
                report['best_model'] = 'ensemble'
                report['best_accuracy'] = float(ensemble_info['accuracy'])
        
        # Save report
        with open(self.output_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_pipeline(self):
        """
        Execute the complete simplified pipeline
        """
        logger.info("Starting simplified classification pipeline...")
        
        # Get features and targets
        X = self.feature_matrix
        y = self.metadata['symptom_encoded'].values
        
        logger.info(f"Original feature matrix shape: {X.shape}")
        
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
        
        # Train-test split (larger test size due to small dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, stratify=y, 
            random_state=self.random_seed
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Training class distribution: {np.bincount(y_train)}")
        logger.info(f"Test class distribution: {np.bincount(y_test)}")
        
        # Feature selection
        X_train_selected, X_test_selected, selected_features, feature_selector = self.select_features(
            X_train, y_train, X_test
        )
        
        # Train models
        results = self.train_models(
            X_train_selected, y_train, X_test_selected, y_test, selected_features
        )
        
        # Create ensemble
        ensemble_pred, ensemble_proba = self.ensemble_predictions(results)
        results['ensemble'] = {
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba,
            'accuracy': accuracy_score(y_test, ensemble_pred)
        }
        
        # Save feature importance for tree-based models
        for model_name in ['random_forest', 'lightgbm']:
            if model_name in results and 'feature_importance' in results[model_name]:
                importance_df = results[model_name]['feature_importance']
                importance_df.to_csv(
                    self.output_dir / f'feature_importance_{model_name}.csv', 
                    index=False
                )
        
        # Create visualizations
        self.create_visualizations(results, y_test)
        
        # Generate report
        report = self.generate_report(results, y_test, selected_features)
        
        # Save best model and preprocessing pipeline
        best_model_name = report['best_model']
        best_model = results[best_model_name]['model'] if best_model_name != 'ensemble' else None
        
        model_save_data = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'scaler': scaler,
            'feature_selector': feature_selector,
            'label_encoder': self.label_encoder,
            'selected_features': selected_features,
            'class_names': list(self.class_names),
            'performance': report['model_performance'][best_model_name]
        }
        
        with open(self.output_dir / 'models/complete_pipeline.pkl', 'wb') as f:
            pickle.dump(model_save_data, f)
        
        logger.info(f"Pipeline complete! Best model: {best_model_name} with accuracy: {report['best_accuracy']:.4f}")
        
        return results, report

def main():
    """
    Main execution
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Microbiome Classification Pipeline')
    parser.add_argument('--features', required=True, help='Path to feature matrix CSV')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV with targets')
    parser.add_argument('--output', default='./model_outputs_microbiome', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_features', type=int, default=100, help='Number of features to select')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SimplifiedMicrobiomeClassifier(
        feature_matrix_path=args.features,
        metadata_path=args.metadata,
        output_dir=args.output,
        random_seed=args.seed,
        n_features=args.n_features
    )
    
    # Run pipeline
    results, report = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("SIMPLIFIED MICROBIOME CLASSIFICATION COMPLETE!")
    print("="*60)
    print(f"Best Model: {report['best_model']}")
    print(f"Best Accuracy: {report['best_accuracy']:.4f}")
    print(f"\nDataset Info:")
    print(f"  - Samples: {report['dataset_info']['n_samples']}")
    print(f"  - Original Features: {report['dataset_info']['n_original_features']}")
    print(f"  - Selected Features: {report['dataset_info']['n_selected_features']}")
    print(f"  - Classes: {report['dataset_info']['n_classes']}")
    print(f"  - Class Distribution: {report['dataset_info']['class_distribution']}")
    
    print(f"\nModel Performance Summary:")
    for model_name, performance in report['model_performance'].items():
        if model_name != 'ensemble':
            print(f"  {model_name}:")
            print(f"    - Test Accuracy: {performance['accuracy']:.3f}")
            print(f"    - CV Accuracy: {performance['cv_accuracy']:.3f} ± {performance['cv_accuracy_std']:.3f}")
            print(f"    - F1-Macro: {performance['f1_macro']:.3f}")
        else:
            print(f"  {model_name}:")
            print(f"    - Test Accuracy: {performance['accuracy']:.3f}")
            print(f"    - F1-Macro: {performance['f1_macro']:.3f}")
    
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()