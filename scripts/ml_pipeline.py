#!/usr/bin/env python3
"""
Machine Learning Pipeline for MPEG-G Challenge Track 1
Cytokine Prediction from Microbiome Data
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
    KFold, TimeSeriesSplit, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicrobiomeCytokineDataset(Dataset):
    """PyTorch Dataset for microbiome-cytokine data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MultiModalTransformer(nn.Module):
    """
    Multi-modal Transformer for microbiome-cytokine prediction
    Handles multiple body sites and temporal information
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int],
                 n_cytokines: int = 62,
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # Body site encoders
        self.site_encoders = nn.ModuleDict({
            site: nn.Linear(dim, hidden_dim)
            for site, dim in input_dims.items()
        })
        
        # Positional encoding for temporal data
        self.temporal_encoding = nn.Embedding(365, hidden_dim)  # Max 365 days
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, n_cytokines)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x_dict, temporal_info=None):
        # Encode each body site
        encoded_sites = []
        for site, encoder in self.site_encoders.items():
            if site in x_dict:
                encoded = encoder(x_dict[site])
                encoded_sites.append(encoded)
        
        # Stack encodings
        x = torch.stack(encoded_sites, dim=1)  # (batch, n_sites, hidden_dim)
        
        # Add temporal encoding if available
        if temporal_info is not None:
            temp_enc = self.temporal_encoding(temporal_info)
            x = x + temp_enc.unsqueeze(1)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Output projection
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class CytokinePredictionPipeline:
    """
    Complete ML pipeline for cytokine prediction
    """
    
    def __init__(self, 
                 feature_matrix_path: str,
                 cytokine_targets_path: str = None,
                 output_dir: str = './model_outputs',
                 random_seed: int = 42):
        
        self.feature_matrix_path = feature_matrix_path
        self.cytokine_targets_path = cytokine_targets_path
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
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and prepare data"""
        logger.info("Loading data...")
        
        # Load feature matrix
        self.feature_matrix = pd.read_csv(self.feature_matrix_path, index_col=0)
        
        # Separate features by type
        self.microbiome_features = [col for col in self.feature_matrix.columns 
                                   if col.startswith('OTU_')]
        self.diversity_features = [col for col in self.feature_matrix.columns 
                                  if col.startswith('Diversity_')]
        self.metadata_features = [col for col in self.feature_matrix.columns 
                                 if col not in self.microbiome_features + self.diversity_features
                                 and col not in ['SubjectID', 'SampleType', 'Date']]
        
        # Load cytokine targets if available
        if self.cytokine_targets_path:
            self.cytokine_data = pd.read_csv(self.cytokine_targets_path, index_col=0)
            self.cytokine_columns = [col for col in self.cytokine_data.columns 
                                    if col.startswith('Cytokine_')]
        else:
            # Generate synthetic targets for testing
            logger.warning("No cytokine data provided. Generating synthetic targets for testing.")
            self.cytokine_columns = [f'Cytokine_{i}' for i in range(62)]
            self.cytokine_data = pd.DataFrame(
                np.random.randn(len(self.feature_matrix), 62),
                index=self.feature_matrix.index,
                columns=self.cytokine_columns
            )
        
        logger.info(f"Loaded {len(self.feature_matrix)} samples with {len(self.feature_matrix.columns)} features")
        logger.info(f"Target: {len(self.cytokine_columns)} cytokines")
        
    def create_body_site_features(self) -> Dict[str, pd.DataFrame]:
        """
        Separate features by body site
        """
        body_sites = ['stool', 'skin', 'nasal', 'oral']
        site_features = {}
        
        for site in body_sites:
            # Filter samples by body site
            site_mask = self.feature_matrix['SampleType'] == site if 'SampleType' in self.feature_matrix else None
            
            if site_mask is not None:
                site_data = self.feature_matrix[site_mask]
                site_features[site] = site_data[self.microbiome_features + self.diversity_features]
            else:
                # If no body site info, use all features
                site_features[site] = self.feature_matrix[self.microbiome_features + self.diversity_features]
                
        return site_features
    
    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering
        """
        X_eng = X.copy()
        
        # 1. Log-transform OTU counts
        otu_cols = [col for col in X.columns if col.startswith('OTU_')]
        X_eng[otu_cols] = np.log1p(X_eng[otu_cols])
        
        # 2. Create ratio features
        if 'Diversity_shannon' in X.columns and 'Diversity_simpson' in X.columns:
            X_eng['diversity_ratio'] = X_eng['Diversity_shannon'] / (X_eng['Diversity_simpson'] + 1e-6)
        
        # 3. Create interaction features for top OTUs
        top_otus = X[otu_cols].sum().nlargest(5).index.tolist()
        for i, otu1 in enumerate(top_otus):
            for otu2 in top_otus[i+1:]:
                X_eng[f'{otu1}_x_{otu2}'] = X_eng[otu1] * X_eng[otu2]
        
        # 4. Aggregate features by taxonomic level (if available)
        # This would require taxonomic annotation of OTUs
        
        return X_eng
    
    def train_baseline_models(self, X_train, y_train, X_test, y_test):
        """
        Train baseline ML models
        """
        results = {}
        
        # 1. Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_seed,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results['random_forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'r2': r2_score(y_test, rf_pred),
            'mae': mean_absolute_error(y_test, rf_pred)
        }
        
        # 2. XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_seed
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        results['xgboost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'r2': r2_score(y_test, xgb_pred),
            'mae': mean_absolute_error(y_test, xgb_pred)
        }
        
        # 3. LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            random_state=self.random_seed
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        
        results['lightgbm'] = {
            'model': lgb_model,
            'predictions': lgb_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, lgb_pred)),
            'r2': r2_score(y_test, lgb_pred),
            'mae': mean_absolute_error(y_test, lgb_pred)
        }
        
        return results
    
    def train_deep_learning_model(self, X_train, y_train, X_test, y_test, 
                                 epochs: int = 100, batch_size: int = 32):
        """
        Train deep learning model
        """
        # Convert to PyTorch datasets
        train_dataset = MicrobiomeCytokineDataset(X_train, y_train)
        test_dataset = MicrobiomeCytokineDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        
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
            
            nn.Linear(128, output_dim)
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Final predictions
        model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                outputs = model(X_batch)
                predictions.append(outputs.numpy())
        
        predictions = np.vstack(predictions)
        
        return {
            'model': model,
            'predictions': predictions,
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'train_history': train_losses,
            'val_history': val_losses
        }
    
    def ensemble_predictions(self, models_dict: Dict):
        """
        Create ensemble predictions
        """
        # Simple averaging ensemble
        all_predictions = []
        weights = []
        
        for model_name, model_info in models_dict.items():
            all_predictions.append(model_info['predictions'])
            # Weight by inverse RMSE
            weights.append(1 / model_info['rmse'])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def feature_importance_analysis(self, model, feature_names):
        """
        Analyze feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top 20
            plt.figure(figsize=(10, 6))
            importance_df.head(20).plot(x='feature', y='importance', kind='barh')
            plt.title('Top 20 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'visualizations/feature_importance.png')
            plt.close()
            
            return importance_df
        
        return None
    
    def generate_submission_report(self, results: Dict):
        """
        Generate final submission report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {},
            'best_model': None,
            'best_rmse': float('inf')
        }
        
        for model_name, model_info in results.items():
            report['model_performance'][model_name] = {
                'rmse': float(model_info['rmse']),
                'r2': float(model_info['r2']),
                'mae': float(model_info['mae'])
            }
            
            if model_info['rmse'] < report['best_rmse']:
                report['best_model'] = model_name
                report['best_rmse'] = float(model_info['rmse'])
        
        # Save report
        with open(self.output_dir / 'submission_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create visualization
        self.create_performance_visualization(results)
        
        return report
    
    def create_performance_visualization(self, results: Dict):
        """
        Create performance comparison visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Prepare data
        models = list(results.keys())
        rmse_scores = [results[m]['rmse'] for m in models]
        r2_scores = [results[m]['r2'] for m in models]
        mae_scores = [results[m]['mae'] for m in models]
        
        # RMSE comparison
        axes[0, 0].bar(models, rmse_scores, color='skyblue')
        axes[0, 0].set_title('RMSE by Model')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R2 comparison
        axes[0, 1].bar(models, r2_scores, color='lightgreen')
        axes[0, 1].set_title('R² Score by Model')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1, 0].bar(models, mae_scores, color='salmon')
        axes[1, 0].set_title('MAE by Model')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Prediction scatter plot for best model
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        predictions = results[best_model]['predictions']
        
        # If we have actual values, plot them
        axes[1, 1].scatter(predictions[:, 0], predictions[:, 1], alpha=0.5)
        axes[1, 1].set_title(f'Sample Predictions ({best_model})')
        axes[1, 1].set_xlabel('Cytokine 1')
        axes[1, 1].set_ylabel('Cytokine 2')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations/model_comparison.png', dpi=300)
        plt.close()
        
    def run_full_pipeline(self):
        """
        Execute complete ML pipeline
        """
        logger.info("Starting ML pipeline...")
        
        # Prepare features
        X = self.feature_matrix[self.microbiome_features + self.diversity_features].fillna(0)
        y = self.cytokine_data[self.cytokine_columns].fillna(0)
        
        # Feature engineering
        logger.info("Performing feature engineering...")
        X = self.feature_engineering(X)
        
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)
        
        # Train-test split
        train_size = int(0.8 * len(X))
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train models
        results = {}
        
        # Baseline models
        baseline_results = self.train_baseline_models(X_train, y_train, X_test, y_test)
        results.update(baseline_results)
        
        # Deep learning model
        logger.info("Training deep learning model...")
        dl_results = self.train_deep_learning_model(X_train, y_train, X_test, y_test)
        results['deep_learning'] = dl_results
        
        # Ensemble
        logger.info("Creating ensemble...")
        ensemble_pred = self.ensemble_predictions(results)
        results['ensemble'] = {
            'predictions': ensemble_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'r2': r2_score(y_test, ensemble_pred),
            'mae': mean_absolute_error(y_test, ensemble_pred)
        }
        
        # Feature importance
        if 'random_forest' in results:
            importance_df = self.feature_importance_analysis(
                results['random_forest']['model'], 
                X.columns
            )
            if importance_df is not None:
                importance_df.to_csv(self.output_dir / 'feature_importance.csv')
        
        # Generate report
        report = self.generate_submission_report(results)
        
        # Save best model
        best_model_name = report['best_model']
        best_model = results[best_model_name]['model']
        
        with open(self.output_dir / 'models/best_model.pkl', 'wb') as f:
            pickle.dump({
                'model': best_model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'model_type': best_model_name,
                'performance': report['model_performance'][best_model_name]
            }, f)
        
        logger.info(f"Pipeline complete! Best model: {best_model_name} with RMSE: {report['best_rmse']:.4f}")
        
        return results, report

def main():
    """
    Main execution
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Pipeline for Cytokine Prediction')
    parser.add_argument('--features', required=True, help='Path to feature matrix CSV')
    parser.add_argument('--cytokines', help='Path to cytokine targets CSV')
    parser.add_argument('--output', default='./model_outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CytokinePredictionPipeline(
        feature_matrix_path=args.features,
        cytokine_targets_path=args.cytokines,
        output_dir=args.output,
        random_seed=args.seed
    )
    
    # Run pipeline
    results, report = pipeline.run_full_pipeline()
    
    print("\n" + "="*50)
    print("ML PIPELINE COMPLETE!")
    print("="*50)
    print(f"Best Model: {report['best_model']}")
    print(f"Best RMSE: {report['best_rmse']:.4f}")
    print(f"\nResults saved to: {args.output}")
    print("\nFor submission, include:")
    print("1. model_outputs/submission_report.json")
    print("2. model_outputs/visualizations/")
    print("3. model_outputs/feature_importance.csv")
    print("4. Your code and documentation")

if __name__ == "__main__":
    main()