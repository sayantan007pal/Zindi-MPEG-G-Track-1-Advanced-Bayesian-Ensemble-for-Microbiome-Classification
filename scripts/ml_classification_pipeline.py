#!/usr/bin/env python3
"""
Classification ML Pipeline for MPEG-G Challenge Track 1
Microbiome-based Symptom Severity Classification
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# ML Libraries
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV, train_test_split
)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MicrobiomeClassificationDataset(Dataset):
    """PyTorch Dataset for microbiome classification data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MicrobiomeClassificationPipeline:
    """
    Complete ML pipeline for microbiome-based symptom severity classification
    """
    
    def __init__(self, 
                 feature_matrix_path: str,
                 metadata_path: str,
                 output_dir: str = './model_outputs_microbiome',
                 random_seed: int = 42):
        
        self.feature_matrix_path = feature_matrix_path
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Set up logging
        log_file = self.output_dir / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        
        # Separate features by type
        self.species_features = [col for col in self.feature_matrix.columns 
                               if col.startswith('species_')]
        self.function_features = [col for col in self.feature_matrix.columns 
                                if col.startswith('function_')]
        self.pathway_features = [col for col in self.feature_matrix.columns 
                               if col.startswith('pathway_')]
        
        # All microbiome features
        self.microbiome_features = self.species_features + self.function_features + self.pathway_features
        
        logger.info(f"Species features: {len(self.species_features)}")
        logger.info(f"Function features: {len(self.function_features)}")
        logger.info(f"Pathway features: {len(self.pathway_features)}")
        logger.info(f"Total microbiome features: {len(self.microbiome_features)}")
        
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
    
    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for microbiome data
        """
        logger.info("Performing feature engineering...")
        X_eng = X.copy()
        
        # Fill any NaN values with 0
        X_eng = X_eng.fillna(0)
        
        # 1. Log-transform abundance data (add small constant to avoid log(0))
        abundance_cols = [col for col in X.columns if col.startswith('species_')]
        X_eng[abundance_cols] = np.log1p(X_eng[abundance_cols].clip(lower=0))
        
        # 2. Create diversity-like metrics using pandas concat for better performance
        new_features = {}
        
        if len(abundance_cols) > 0:
            # Shannon-like diversity
            abundance_data = X_eng[abundance_cols].values + 1e-10  # avoid log(0)
            row_sums = abundance_data.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1e-10, row_sums)  # avoid division by 0
            proportions = abundance_data / row_sums
            shannon_diversity = -np.sum(proportions * np.log(proportions + 1e-10), axis=1)
            new_features['calculated_shannon_diversity'] = shannon_diversity
            
            # Richness (number of non-zero species)
            new_features['species_richness'] = (X_eng[abundance_cols] > 0).sum(axis=1).values
            
            # Simpson-like index
            simpson_index = np.sum(proportions ** 2, axis=1)
            new_features['calculated_simpson_index'] = simpson_index
            
            # Evenness (handle division by zero)
            richness_log = np.log(new_features['species_richness'] + 1e-10)
            evenness = shannon_diversity / np.where(richness_log == 0, 1e-10, richness_log)
            new_features['calculated_evenness'] = evenness
        
        # 3. Create ratio features for top species (limit to avoid too many features)
        if len(abundance_cols) >= 3:
            top_species = X[abundance_cols].sum().nlargest(3).index.tolist()
            for i, species1 in enumerate(top_species):
                for species2 in top_species[i+1:]:
                    ratio_name = f'ratio_{i}_{i+1}'  # Use simpler names
                    denominator = X_eng[species2] + 1e-10
                    new_features[ratio_name] = (X_eng[species1] + 1e-10) / denominator
        
        # 4. Aggregate features by taxonomic groups (simplified)
        bacteroides_cols = [col for col in abundance_cols if 'Bacteroides' in col]
        if bacteroides_cols:
            new_features['bacteroides_total'] = X_eng[bacteroides_cols].sum(axis=1).values
            
        bifidobacterium_cols = [col for col in abundance_cols if 'Bifidobacterium' in col]
        if bifidobacterium_cols:
            new_features['bifidobacterium_total'] = X_eng[bifidobacterium_cols].sum(axis=1).values
            
        clostridium_cols = [col for col in abundance_cols if 'Clostridium' in col]
        if clostridium_cols:
            new_features['clostridium_total'] = X_eng[clostridium_cols].sum(axis=1).values
        
        # Add new features using concat for better performance
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=X_eng.index)
            X_eng = pd.concat([X_eng, new_features_df], axis=1)
        
        # Final check for any remaining NaN/inf values
        X_eng = X_eng.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.info(f"Feature engineering complete. New shape: {X_eng.shape}")
        return X_eng
    
    def train_baseline_models(self, X_train, y_train, X_test, y_test, cv_folds=5):
        """
        Train baseline ML models for classification
        """
        results = {}
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        
        # 1. Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_seed,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Cross-validation
        cv_scores = cross_validate(rf_model, X_train, y_train, cv=cv, 
                                 scoring=['accuracy', 'f1_macro', 'roc_auc_ovr'])
        
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)
        
        results['random_forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'probabilities': rf_proba,
            'accuracy': accuracy_score(y_test, rf_pred),
            'cv_accuracy': cv_scores['test_accuracy'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_f1_macro': cv_scores['test_f1_macro'].mean(),
            'cv_f1_macro_std': cv_scores['test_f1_macro'].std(),
            'cv_auc': cv_scores['test_roc_auc_ovr'].mean(),
            'cv_auc_std': cv_scores['test_roc_auc_ovr'].std()
        }
        
        # 2. XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_seed,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        cv_scores = cross_validate(xgb_model, X_train, y_train, cv=cv, 
                                 scoring=['accuracy', 'f1_macro', 'roc_auc_ovr'])
        
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)
        
        results['xgboost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'probabilities': xgb_proba,
            'accuracy': accuracy_score(y_test, xgb_pred),
            'cv_accuracy': cv_scores['test_accuracy'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_f1_macro': cv_scores['test_f1_macro'].mean(),
            'cv_f1_macro_std': cv_scores['test_f1_macro'].std(),
            'cv_auc': cv_scores['test_roc_auc_ovr'].mean(),
            'cv_auc_std': cv_scores['test_roc_auc_ovr'].std()
        }
        
        # 3. LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            random_state=self.random_seed,
            class_weight='balanced',
            verbosity=-1
        )
        
        cv_scores = cross_validate(lgb_model, X_train, y_train, cv=cv, 
                                 scoring=['accuracy', 'f1_macro', 'roc_auc_ovr'])
        
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)
        
        results['lightgbm'] = {
            'model': lgb_model,
            'predictions': lgb_pred,
            'probabilities': lgb_proba,
            'accuracy': accuracy_score(y_test, lgb_pred),
            'cv_accuracy': cv_scores['test_accuracy'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_f1_macro': cv_scores['test_f1_macro'].mean(),
            'cv_f1_macro_std': cv_scores['test_f1_macro'].std(),
            'cv_auc': cv_scores['test_roc_auc_ovr'].mean(),
            'cv_auc_std': cv_scores['test_roc_auc_ovr'].std()
        }
        
        # 4. Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=self.random_seed,
            max_iter=1000,
            class_weight='balanced',
            multi_class='ovr'
        )
        
        cv_scores = cross_validate(lr_model, X_train, y_train, cv=cv, 
                                 scoring=['accuracy', 'f1_macro', 'roc_auc_ovr'])
        
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_proba = lr_model.predict_proba(X_test)
        
        results['logistic_regression'] = {
            'model': lr_model,
            'predictions': lr_pred,
            'probabilities': lr_proba,
            'accuracy': accuracy_score(y_test, lr_pred),
            'cv_accuracy': cv_scores['test_accuracy'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_f1_macro': cv_scores['test_f1_macro'].mean(),
            'cv_f1_macro_std': cv_scores['test_f1_macro'].std(),
            'cv_auc': cv_scores['test_roc_auc_ovr'].mean(),
            'cv_auc_std': cv_scores['test_roc_auc_ovr'].std()
        }
        
        return results
    
    def train_deep_learning_model(self, X_train, y_train, X_test, y_test, 
                                 epochs: int = 100, batch_size: int = 16):
        """
        Train deep learning model for classification
        """
        logger.info("Training deep learning model...")
        
        # Convert to PyTorch datasets
        train_dataset = MicrobiomeClassificationDataset(X_train, y_train)
        test_dataset = MicrobiomeClassificationDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = X_train.shape[1]
        
        model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, self.n_classes)
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()
            
            # Validate
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {avg_val_loss:.4f}, Train Acc = {train_acc:.4f}, "
                          f"Val Acc = {val_acc:.4f}")
        
        # Final predictions
        model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for X_batch, _ in test_loader:
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                predictions.extend(preds.numpy())
                probabilities.extend(probs.numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        return {
            'model': model,
            'predictions': predictions,
            'probabilities': probabilities,
            'accuracy': accuracy_score(y_test, predictions),
            'train_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }
        }
    
    def ensemble_predictions(self, models_dict: Dict):
        """
        Create ensemble predictions using voting
        """
        logger.info("Creating ensemble predictions...")
        
        # Collect predictions and probabilities
        all_predictions = []
        all_probabilities = []
        weights = []
        
        for model_name, model_info in models_dict.items():
            if model_name != 'ensemble':  # avoid recursive ensemble
                all_predictions.append(model_info['predictions'])
                all_probabilities.append(model_info['probabilities'])
                # Weight by cross-validation accuracy if available
                weight = model_info.get('cv_accuracy', model_info.get('accuracy', 0.5))
                weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average of probabilities
        ensemble_proba = np.average(all_probabilities, axis=0, weights=weights)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, ensemble_proba
    
    def feature_importance_analysis(self, model, feature_names, model_name):
        """
        Analyze feature importance for tree-based models
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save to CSV
            importance_df.to_csv(self.output_dir / f'feature_importance_{model_name}.csv', index=False)
            
            # Plot top 20
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 20 Feature Importances - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.output_dir / f'visualizations/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return importance_df
        
        return None
    
    def create_confusion_matrix_plot(self, y_true, y_pred, model_name):
        """
        Create confusion matrix visualization
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'visualizations/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_classification_report(self, results: Dict, y_test):
        """
        Generate comprehensive classification report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {},
            'best_model': None,
            'best_accuracy': 0,
            'dataset_info': {
                'n_samples': len(self.feature_matrix),
                'n_features': len(self.microbiome_features),
                'n_classes': self.n_classes,
                'class_names': list(self.class_names),
                'class_distribution': self.metadata['symptom'].value_counts().to_dict()
            }
        }
        
        for model_name, model_info in results.items():
            if model_name != 'ensemble':
                # Calculate additional metrics
                y_pred = model_info['predictions']
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='macro', zero_division=0
                )
                
                # Classification report
                class_report = classification_report(
                    y_test, y_pred, 
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0
                )
                
                report['model_performance'][model_name] = {
                    'accuracy': float(model_info['accuracy']),
                    'cv_accuracy': float(model_info.get('cv_accuracy', 0)),
                    'cv_accuracy_std': float(model_info.get('cv_accuracy_std', 0)),
                    'precision_macro': float(precision),
                    'recall_macro': float(recall),
                    'f1_macro': float(f1),
                    'cv_f1_macro': float(model_info.get('cv_f1_macro', 0)),
                    'cv_auc': float(model_info.get('cv_auc', 0)),
                    'per_class_metrics': class_report
                }
                
                if model_info['accuracy'] > report['best_accuracy']:
                    report['best_model'] = model_name
                    report['best_accuracy'] = float(model_info['accuracy'])
                
                # Create confusion matrix
                self.create_confusion_matrix_plot(y_test, y_pred, model_name)
        
        # Handle ensemble results
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
            
            self.create_confusion_matrix_plot(y_test, y_pred, 'ensemble')
        
        # Save report
        with open(self.output_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create performance visualization
        self.create_performance_visualization(results)
        
        return report
    
    def create_performance_visualization(self, results: Dict):
        """
        Create comprehensive performance comparison visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        models = [name for name in results.keys() if name != 'ensemble']
        if 'ensemble' in results:
            models.append('ensemble')
        
        accuracies = [results[m]['accuracy'] for m in models]
        cv_accuracies = [results[m].get('cv_accuracy', 0) for m in models if m != 'ensemble']
        cv_f1_scores = [results[m].get('cv_f1_macro', 0) for m in models if m != 'ensemble']
        cv_auc_scores = [results[m].get('cv_auc', 0) for m in models if m != 'ensemble']
        
        # Test Accuracy comparison
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Test Accuracy by Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim([0, 1])
        
        # Cross-validation metrics (excluding ensemble)
        cv_models = [m for m in models if m != 'ensemble']
        if cv_models:
            x_pos = np.arange(len(cv_models))
            width = 0.25
            
            axes[0, 1].bar(x_pos - width, cv_accuracies, width, label='CV Accuracy', color='lightgreen')
            axes[0, 1].bar(x_pos, cv_f1_scores, width, label='CV F1-Macro', color='salmon')
            axes[0, 1].bar(x_pos + width, cv_auc_scores, width, label='CV AUC', color='gold')
            
            axes[0, 1].set_title('Cross-Validation Metrics')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(cv_models, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].set_ylim([0, 1])
        
        # Training history for deep learning model
        if 'deep_learning' in results and 'train_history' in results['deep_learning']:
            history = results['deep_learning']['train_history']
            epochs = range(len(history['train_losses']))
            
            axes[1, 0].plot(epochs, history['train_losses'], label='Train Loss', color='blue')
            axes[1, 0].plot(epochs, history['val_losses'], label='Val Loss', color='red')
            axes[1, 0].set_title('Deep Learning Training History - Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            
            ax_twin = axes[1, 0].twinx()
            ax_twin.plot(epochs, history['train_accuracies'], label='Train Acc', color='lightblue', linestyle='--')
            ax_twin.plot(epochs, history['val_accuracies'], label='Val Acc', color='lightcoral', linestyle='--')
            ax_twin.set_ylabel('Accuracy')
            ax_twin.legend(loc='center right')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Deep Learning\nTraining History', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12)
            axes[1, 0].set_title('Deep Learning Training History')
        
        # Model comparison radar chart would go here if we had more metrics
        # For now, show a summary table
        axes[1, 1].axis('off')
        
        # Create summary table
        summary_data = []
        for model in models:
            if model in results:
                summary_data.append([
                    model,
                    f"{results[model]['accuracy']:.3f}",
                    f"{results[model].get('cv_accuracy', 0):.3f}" if model != 'ensemble' else 'N/A',
                    f"{results[model].get('cv_f1_macro', 0):.3f}" if model != 'ensemble' else 'N/A'
                ])
        
        table = axes[1, 1].table(cellText=summary_data,
                               colLabels=['Model', 'Test Acc', 'CV Acc', 'CV F1'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Model Performance Summary')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_pipeline(self):
        """
        Execute complete classification pipeline
        """
        logger.info("Starting classification pipeline...")
        
        # Prepare features and targets
        X = self.feature_matrix[self.microbiome_features].fillna(0)
        y = self.metadata['symptom_encoded'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Feature engineering
        X_eng = self.feature_engineering(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_eng)
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, 
            random_state=self.random_seed
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Training class distribution: {np.bincount(y_train)}")
        logger.info(f"Test class distribution: {np.bincount(y_test)}")
        
        # Train models
        results = {}
        
        # Baseline models
        logger.info("Training baseline models...")
        baseline_results = self.train_baseline_models(X_train, y_train, X_test, y_test)
        results.update(baseline_results)
        
        # Deep learning model
        dl_results = self.train_deep_learning_model(X_train, y_train, X_test, y_test)
        results['deep_learning'] = dl_results
        
        # Ensemble
        ensemble_pred, ensemble_proba = self.ensemble_predictions(results)
        results['ensemble'] = {
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba,
            'accuracy': accuracy_score(y_test, ensemble_pred)
        }
        
        # Feature importance analysis
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            if model_name in results:
                importance_df = self.feature_importance_analysis(
                    results[model_name]['model'], 
                    X_eng.columns,
                    model_name
                )
        
        # Generate comprehensive report
        report = self.generate_classification_report(results, y_test)
        
        # Save best model
        best_model_name = report['best_model']
        best_model = results[best_model_name]['model']
        
        model_save_data = {
            'model': best_model,
            'scaler': scaler,
            'label_encoder': self.label_encoder,
            'model_type': best_model_name,
            'performance': report['model_performance'][best_model_name],
            'feature_names': list(X_eng.columns),
            'class_names': list(self.class_names)
        }
        
        with open(self.output_dir / 'models/best_model.pkl', 'wb') as f:
            pickle.dump(model_save_data, f)
        
        logger.info(f"Pipeline complete! Best model: {best_model_name} with accuracy: {report['best_accuracy']:.4f}")
        
        return results, report

def main():
    """
    Main execution
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Classification ML Pipeline for Microbiome Data')
    parser.add_argument('--features', required=True, help='Path to feature matrix CSV')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV with targets')
    parser.add_argument('--output', default='./model_outputs_microbiome', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MicrobiomeClassificationPipeline(
        feature_matrix_path=args.features,
        metadata_path=args.metadata,
        output_dir=args.output,
        random_seed=args.seed
    )
    
    # Run pipeline
    results, report = pipeline.run_full_pipeline()
    
    print("\n" + "="*60)
    print("MICROBIOME CLASSIFICATION PIPELINE COMPLETE!")
    print("="*60)
    print(f"Best Model: {report['best_model']}")
    print(f"Best Accuracy: {report['best_accuracy']:.4f}")
    print(f"\nDataset Info:")
    print(f"  - Samples: {report['dataset_info']['n_samples']}")
    print(f"  - Features: {report['dataset_info']['n_features']}")
    print(f"  - Classes: {report['dataset_info']['n_classes']}")
    print(f"  - Class Distribution: {report['dataset_info']['class_distribution']}")
    print(f"\nResults saved to: {args.output}")
    print("\nKey outputs:")
    print("1. classification_report.json - Comprehensive performance metrics")
    print("2. visualizations/ - Performance plots and confusion matrices")
    print("3. feature_importance_*.csv - Feature importance rankings")
    print("4. models/best_model.pkl - Best trained model")

if __name__ == "__main__":
    main()