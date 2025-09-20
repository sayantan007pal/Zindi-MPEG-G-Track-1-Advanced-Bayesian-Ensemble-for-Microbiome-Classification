#!/usr/bin/env python3
"""
MPEG-G Track 1 Submission Model - Bayesian Optimized Ensemble

This script implements the final optimized model for MPEG-G Track 1 submission.
Based on comprehensive Bayesian optimization achieving 95.0% CV accuracy.

Model Configuration:
- Bayesian Optimized Ensemble (Random Forest + Gradient Boosting + Logistic Regression)
- 10 selected features from enhanced feature set
- Optimal ensemble weights: [52.2%, 39.0%, 8.7%]
- 95% confidence interval: [82.1%, 100.0%]

Author: Final Optimization Pipeline
Date: September 20, 2025
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, List
import logging

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

class MPEGTrack1SubmissionModel:
    """
    Final submission model for MPEG-G Track 1 challenge.
    Implements the Bayesian optimized ensemble with selected features.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        self.selected_features = [
            'change_function_K03750',
            'change_function_K02588', 
            'change_species_GUT_GENOME234915',
            'pca_component_2',
            'change_species_GUT_GENOME091092',
            'temporal_var_species_GUT_GENOME002690',
            'change_species_Blautia schinkii',
            'pca_component_1',
            'stability_function_K07466',
            'change_function_K03484'
        ]
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for model operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def create_optimized_model(self) -> VotingClassifier:
        """
        Create the Bayesian optimized ensemble model.
        
        Returns:
            VotingClassifier: The optimized ensemble model
        """
        self.logger.info("Creating Bayesian optimized ensemble model...")
        
        # Random Forest with optimized parameters (52.2% weight)
        rf_optimized = RandomForestClassifier(
            n_estimators=500,
            max_depth=1,
            min_samples_split=8,
            min_samples_leaf=2,
            max_features=1.0,
            criterion='gini',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting with optimized parameters (39.0% weight)
        gb_optimized = GradientBoostingClassifier(
            n_estimators=290,
            learning_rate=0.08076060788613365,
            max_depth=10,
            min_samples_split=6,
            subsample=0.9345745243582918,
            random_state=self.random_state
        )
        
        # Logistic Regression (8.7% weight)
        lr_optimized = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Create ensemble with optimized weights
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_optimized),
                ('gb', gb_optimized),
                ('lr', lr_optimized)
            ],
            voting='soft',
            weights=[0.522334984317205, 0.3901717053591802, 0.0874933103236147]
        )
        
        self.logger.info("Model created successfully with optimal configuration")
        return ensemble
        
    def load_and_prepare_data(self, features_path: str, metadata_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and prepare data with feature selection.
        
        Args:
            features_path: Path to features CSV file
            metadata_path: Path to metadata CSV file
            
        Returns:
            Tuple of (selected_features_df, encoded_labels)
        """
        self.logger.info("Loading and preparing data...")
        
        # Load data
        X = pd.read_csv(features_path, index_col=0)
        metadata = pd.read_csv(metadata_path, index_col=0)
        y = metadata['symptom']
        
        self.logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Select optimized features
        # Check if all selected features are available
        missing_features = [f for f in self.selected_features if f not in X.columns]
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            # Use available features only
            available_features = [f for f in self.selected_features if f in X.columns]
            self.selected_features = available_features
        
        X_selected = X[self.selected_features]
        self.logger.info(f"Selected {len(self.selected_features)} features for training")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.logger.info(f"Classes: {self.label_encoder.classes_}")
        return X_selected, y_encoded
        
    def train(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Train the optimized model.
        
        Args:
            X: Feature matrix
            y: Target labels (encoded)
        """
        self.logger.info("Training Bayesian optimized ensemble...")
        
        # Create and train model
        self.model = self.create_optimized_model()
        self.model.fit(X, y)
        
        self.logger.info("Training completed successfully")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels (original labels, not encoded)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Ensure we have the right features
        X_selected = X[self.selected_features]
        
        # Make predictions
        y_pred_encoded = self.model.predict(X_selected)
        
        # Decode predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        X_selected = X[self.selected_features]
        return self.model.predict_proba(X_selected)
        
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, cv_folds: int = 5) -> dict:
        """
        Evaluate model performance using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Evaluating model performance...")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        # Train and evaluate on full dataset
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        full_accuracy = accuracy_score(y, y_pred)
        
        # Generate classification report
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_true_labels = self.label_encoder.inverse_transform(y)
        class_report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
        
        results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'full_dataset_accuracy': float(full_accuracy),
            'classification_report': class_report
        }
        
        self.logger.info(f"CV Accuracy: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        return results
        
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
            
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'selected_features': self.selected_features,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.selected_features = model_data['selected_features']
        self.random_state = model_data['random_state']
        
        self.logger.info(f"Model loaded from {path}")

def main():
    """
    Main function to train and evaluate the submission model.
    """
    print("="*80)
    print("MPEG-G TRACK 1 SUBMISSION MODEL")
    print("Bayesian Optimized Ensemble - Final Configuration")
    print("="*80)
    
    # Initialize model
    submission_model = MPEGTrack1SubmissionModel(random_state=42)
    
    # Prepare data paths
    features_path = "enhanced_features/enhanced_features_final.csv"
    metadata_path = "enhanced_features/enhanced_metadata_final.csv"
    
    # Check if files exist, fallback to processed data
    if not Path(features_path).exists():
        print("Enhanced features not found, using processed microbiome data...")
        features_path = "processed_data/microbiome_features_processed.csv"
        metadata_path = "processed_data/microbiome_metadata_processed.csv"
    
    try:
        # Load and prepare data
        X, y = submission_model.load_and_prepare_data(features_path, metadata_path)
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Selected features: {len(submission_model.selected_features)}")
        
        # Train model
        submission_model.train(X, y)
        print("Model training completed")
        
        # Evaluate performance
        results = submission_model.evaluate(X, y)
        
        print(f"\nPERFORMANCE EVALUATION:")
        print(f"Cross-Validation Accuracy: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        print(f"Full Dataset Accuracy: {results['full_dataset_accuracy']:.3f}")
        
        print(f"\nCLASS-WISE PERFORMANCE:")
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                print(f"  {class_name}: F1={metrics['f1-score']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
        
        # Save the trained model
        output_dir = Path("final_optimization_outputs")
        output_dir.mkdir(exist_ok=True)
        model_path = output_dir / "submission_model.pkl"
        submission_model.save_model(str(model_path))
        
        print(f"\nModel saved to: {model_path}")
        print("\nSUBMISSION MODEL READY!")
        print("="*80)
        
        return submission_model, results
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    model, results = main()