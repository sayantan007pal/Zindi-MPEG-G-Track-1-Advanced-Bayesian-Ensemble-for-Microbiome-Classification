#!/usr/bin/env python3
"""
Advanced Ensemble Pipeline for MPEG-G Microbiome Challenge Track 1
================================================================

Implements comprehensive ensemble methods to achieve 90%+ accuracy:
- Stacking ensemble with cross-validation
- Voting ensemble (hard, soft, weighted)
- Bagging with feature subsets
- Dynamic ensemble selection
- Diversity-based pruning
- Confidence-weighted predictions

Based on enhanced temporal features (20 subjects × 69 features)
Target: Improve from 85% baseline to 90%+ accuracy

Author: Sayantan Pal
Date: 2025-09-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict, 
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, BaggingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support, log_loss
)
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not available")

import json
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiversityMetrics:
    """Calculate diversity metrics for ensemble pruning"""
    
    @staticmethod
    def pairwise_diversity(pred1, pred2):
        """Calculate pairwise diversity between two prediction sets"""
        return np.mean(pred1 != pred2)
    
    @staticmethod
    def q_statistic(pred1, pred2, y_true):
        """Calculate Q-statistic for diversity measurement"""
        n11 = np.sum((pred1 == y_true) & (pred2 == y_true))
        n10 = np.sum((pred1 == y_true) & (pred2 != y_true))
        n01 = np.sum((pred1 != y_true) & (pred2 == y_true))
        n00 = np.sum((pred1 != y_true) & (pred2 != y_true))
        
        numerator = n11 * n00 - n01 * n10
        denominator = n11 * n00 + n01 * n10
        
        return numerator / (denominator + 1e-10)
    
    @staticmethod
    def correlation_coefficient(pred1, pred2):
        """Calculate correlation coefficient between predictions"""
        return np.corrcoef(pred1, pred2)[0, 1]


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced stacking ensemble with cross-validation"""
    
    def __init__(self, base_models, meta_model, cv=5, use_probas=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.use_probas = use_probas
        self.fitted_base_models = []
        self.fitted_meta_model = None
        
    def fit(self, X, y):
        """Fit the stacking ensemble"""
        # Generate cross-validation predictions
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        meta_features = []
        
        for name, model in self.base_models:
            logger.info(f"Training base model: {name}")
            
            if self.use_probas and hasattr(model, 'predict_proba'):
                cv_preds = cross_val_predict(
                    model, X, y, cv=skf, method='predict_proba'
                )
                meta_features.append(cv_preds)
            else:
                cv_preds = cross_val_predict(model, X, y, cv=skf)
                meta_features.append(cv_preds.reshape(-1, 1))
        
        # Concatenate meta features
        if self.use_probas:
            meta_X = np.concatenate(meta_features, axis=1)
        else:
            meta_X = np.concatenate(meta_features, axis=1)
        
        # Fit meta model
        self.fitted_meta_model = self.meta_model.fit(meta_X, y)
        
        # Fit base models on full data
        self.fitted_base_models = []
        for name, model in self.base_models:
            fitted_model = model.fit(X, y)
            self.fitted_base_models.append((name, fitted_model))
        
        return self
    
    def predict(self, X):
        """Make predictions using the stacking ensemble"""
        meta_features = []
        
        for name, model in self.fitted_base_models:
            if self.use_probas and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
                meta_features.append(preds)
            else:
                preds = model.predict(X)
                meta_features.append(preds.reshape(-1, 1))
        
        if self.use_probas:
            meta_X = np.concatenate(meta_features, axis=1)
        else:
            meta_X = np.concatenate(meta_features, axis=1)
        
        return self.fitted_meta_model.predict(meta_X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self.fitted_meta_model, 'predict_proba'):
            meta_features = []
            
            for name, model in self.fitted_base_models:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)
                    meta_features.append(preds)
                else:
                    preds = model.predict(X)
                    meta_features.append(preds.reshape(-1, 1))
            
            if self.use_probas:
                meta_X = np.concatenate(meta_features, axis=1)
            else:
                meta_X = np.concatenate(meta_features, axis=1)
            
            return self.fitted_meta_model.predict_proba(meta_X)
        else:
            return None


class DynamicEnsembleSelector(BaseEstimator, ClassifierMixin):
    """Dynamic ensemble selection based on sample characteristics"""
    
    def __init__(self, base_models, selection_strategy='accuracy', k_neighbors=3):
        self.base_models = base_models
        self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors
        self.fitted_models = []
        self.validation_accuracy = {}
        
    def fit(self, X, y):
        """Fit models and calculate validation accuracy"""
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        for name, model in self.base_models:
            fitted_model = model.fit(X_train, y_train)
            val_pred = fitted_model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            
            self.fitted_models.append((name, fitted_model))
            self.validation_accuracy[name] = val_acc
            
        # Store validation data for neighbor-based selection
        self.X_val = X_val
        self.y_val = y_val
        
        return self
    
    def predict(self, X):
        """Predict using dynamic selection"""
        predictions = []
        
        for i in range(len(X)):
            sample = X[i:i+1]
            
            if self.selection_strategy == 'accuracy':
                # Select best performing model
                best_model_name = max(self.validation_accuracy, 
                                    key=self.validation_accuracy.get)
                selected_model = next(
                    model for name, model in self.fitted_models 
                    if name == best_model_name
                )
            
            elif self.selection_strategy == 'local':
                # Select based on local accuracy (k-nearest neighbors)
                distances = np.linalg.norm(self.X_val - sample, axis=1)
                nearest_indices = np.argsort(distances)[:self.k_neighbors]
                
                best_local_acc = 0
                selected_model = self.fitted_models[0][1]
                
                for name, model in self.fitted_models:
                    local_pred = model.predict(self.X_val[nearest_indices])
                    local_acc = accuracy_score(
                        self.y_val[nearest_indices], local_pred
                    )
                    
                    if local_acc > best_local_acc:
                        best_local_acc = local_acc
                        selected_model = model
            
            pred = selected_model.predict(sample)[0]
            predictions.append(pred)
        
        return np.array(predictions)


class AdvancedEnsemblePipeline:
    """Advanced ensemble pipeline for MPEG-G Challenge"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.ensembles = {}
        
        # Set up base models
        self._setup_base_models()
        
    def _setup_base_models(self):
        """Setup base models with optimized hyperparameters"""
        
        # Core models with optimized parameters
        self.base_models = [
            ('lr', LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear', 
                max_iter=1000, random_state=self.random_state
            )),
            ('lr_l1', LogisticRegression(
                C=0.1, penalty='l1', solver='liblinear',
                max_iter=1000, random_state=self.random_state
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state
            )),
            ('rf_deep', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=3,
                min_samples_leaf=1, random_state=self.random_state
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=100, max_depth=8, min_samples_split=4,
                random_state=self.random_state
            )),
            ('svm_rbf', SVC(
                C=1.0, kernel='rbf', gamma='scale', probability=True,
                random_state=self.random_state
            )),
            ('svm_linear', SVC(
                C=0.1, kernel='linear', probability=True,
                random_state=self.random_state
            )),
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance')),
            ('ridge', RidgeClassifier(alpha=1.0, random_state=self.random_state)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(50, 25), alpha=0.01, max_iter=1000,
                random_state=self.random_state
            )),
            ('ada', AdaBoostClassifier(
                n_estimators=50, learning_rate=1.0,
                random_state=self.random_state
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=self.random_state
            ))
        ]
        
        # Add XGBoost and LightGBM if available
        if HAS_XGB:
            self.base_models.append(
                ('xgb', xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=self.random_state, eval_metric='logloss'
                ))
            )
        
        if HAS_LGB:
            self.base_models.append(
                ('lgb', lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=self.random_state, verbose=-1
                ))
            )
    
    def evaluate_base_models(self, X, y, cv=5):
        """Evaluate individual base models"""
        logger.info("Evaluating base models...")
        
        results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.base_models:
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
                results[name] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std(),
                    'scores': scores.tolist()
                }
                logger.info(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                results[name] = {
                    'mean_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'scores': [0.0] * cv
                }
        
        self.results['base_models'] = results
        return results
    
    def create_voting_ensembles(self, X, y):
        """Create voting ensembles"""
        logger.info("Creating voting ensembles...")
        
        # Select top performing models for voting
        base_results = self.results.get('base_models', {})
        top_models = sorted(
            base_results.items(), 
            key=lambda x: x[1]['mean_accuracy'], 
            reverse=True
        )[:7]  # Top 7 models
        
        voting_models = [(name, dict(self.base_models)[name]) for name, _ in top_models]
        
        # Hard voting
        hard_voting = VotingClassifier(
            estimators=voting_models,
            voting='hard'
        )
        
        # Soft voting
        soft_voting = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        
        # Weighted voting (weights based on performance)
        weights = [results['mean_accuracy'] for _, results in top_models]
        weighted_voting = VotingClassifier(
            estimators=voting_models,
            voting='soft',
            weights=weights
        )
        
        self.ensembles['hard_voting'] = hard_voting
        self.ensembles['soft_voting'] = soft_voting
        self.ensembles['weighted_voting'] = weighted_voting
        
        return {
            'hard_voting': hard_voting,
            'soft_voting': soft_voting,
            'weighted_voting': weighted_voting
        }
    
    def create_stacking_ensemble(self, X, y):
        """Create stacking ensemble"""
        logger.info("Creating stacking ensemble...")
        
        # Select diverse base models
        base_results = self.results.get('base_models', {})
        selected_models = []
        
        # Select top models from different families
        model_families = {
            'linear': ['lr', 'lr_l1', 'ridge'],
            'tree': ['rf', 'rf_deep', 'et', 'gb'],
            'svm': ['svm_rbf', 'svm_linear'],
            'other': ['nb', 'knn', 'mlp', 'ada']
        }
        
        if HAS_XGB:
            model_families['tree'].append('xgb')
        if HAS_LGB:
            model_families['tree'].append('lgb')
        
        for family, models in model_families.items():
            family_results = {
                name: base_results.get(name, {'mean_accuracy': 0})
                for name in models
                if name in base_results
            }
            if family_results:
                best_in_family = max(
                    family_results.items(),
                    key=lambda x: x[1]['mean_accuracy']
                )[0]
                selected_models.append((best_in_family, dict(self.base_models)[best_in_family]))
        
        # Meta-learner
        meta_model = LogisticRegression(
            C=0.1, random_state=self.random_state, max_iter=1000
        )
        
        stacking = StackingEnsemble(
            base_models=selected_models,
            meta_model=meta_model,
            cv=5,
            use_probas=True
        )
        
        self.ensembles['stacking'] = stacking
        return stacking
    
    def create_bagging_ensembles(self, X, y):
        """Create bagging ensembles with feature subsets"""
        logger.info("Creating bagging ensembles...")
        
        # Feature bagging with Random Forest
        feature_bagging_rf = BaggingClassifier(
            estimator=RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=self.random_state
            ),
            n_estimators=10,
            max_features=0.8,
            max_samples=0.8,
            random_state=self.random_state
        )
        
        # Feature bagging with Logistic Regression
        feature_bagging_lr = BaggingClassifier(
            estimator=LogisticRegression(
                C=1.0, random_state=self.random_state, max_iter=1000
            ),
            n_estimators=10,
            max_features=0.7,
            max_samples=0.9,
            random_state=self.random_state
        )
        
        # Random subspace method
        random_subspace = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=8, random_state=self.random_state
            ),
            n_estimators=15,
            max_features=0.6,
            bootstrap=False,
            random_state=self.random_state
        )
        
        self.ensembles['feature_bagging_rf'] = feature_bagging_rf
        self.ensembles['feature_bagging_lr'] = feature_bagging_lr
        self.ensembles['random_subspace'] = random_subspace
        
        return {
            'feature_bagging_rf': feature_bagging_rf,
            'feature_bagging_lr': feature_bagging_lr,
            'random_subspace': random_subspace
        }
    
    def create_dynamic_ensemble(self, X, y):
        """Create dynamic ensemble selector"""
        logger.info("Creating dynamic ensemble...")
        
        # Select top 5 diverse models
        base_results = self.results.get('base_models', {})
        top_models = sorted(
            base_results.items(),
            key=lambda x: x[1]['mean_accuracy'],
            reverse=True
        )[:5]
        
        dynamic_models = [(name, dict(self.base_models)[name]) for name, _ in top_models]
        
        dynamic_ensemble = DynamicEnsembleSelector(
            base_models=dynamic_models,
            selection_strategy='local',
            k_neighbors=3
        )
        
        self.ensembles['dynamic'] = dynamic_ensemble
        return dynamic_ensemble
    
    def diversity_based_pruning(self, X, y, max_models=8):
        """Prune ensemble using diversity metrics"""
        logger.info("Performing diversity-based pruning...")
        
        # Get cross-validation predictions for all models
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        model_predictions = {}
        
        for name, model in self.base_models:
            try:
                cv_preds = cross_val_predict(model, X, y, cv=skf)
                model_predictions[name] = cv_preds
            except:
                continue
        
        # Calculate diversity matrix
        model_names = list(model_predictions.keys())
        diversity_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i != j:
                    diversity = DiversityMetrics.pairwise_diversity(
                        model_predictions[name1], model_predictions[name2]
                    )
                    diversity_matrix[i, j] = diversity
        
        # Select diverse subset
        selected_indices = [0]  # Start with first model
        
        while len(selected_indices) < max_models and len(selected_indices) < len(model_names):
            remaining_indices = [i for i in range(len(model_names)) if i not in selected_indices]
            
            best_candidate = None
            best_avg_diversity = 0
            
            for candidate in remaining_indices:
                avg_diversity = np.mean([
                    diversity_matrix[candidate, selected]
                    for selected in selected_indices
                ])
                
                if avg_diversity > best_avg_diversity:
                    best_avg_diversity = avg_diversity
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
            else:
                break
        
        # Create pruned ensemble
        selected_models = [
            (model_names[i], dict(self.base_models)[model_names[i]])
            for i in selected_indices
        ]
        
        pruned_ensemble = VotingClassifier(
            estimators=selected_models,
            voting='soft'
        )
        
        self.ensembles['diversity_pruned'] = pruned_ensemble
        return pruned_ensemble, selected_models
    
    def evaluate_ensembles(self, X, y, cv=5):
        """Evaluate all ensemble methods"""
        logger.info("Evaluating ensemble methods...")
        
        ensemble_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, ensemble in self.ensembles.items():
            try:
                scores = cross_val_score(ensemble, X, y, cv=skf, scoring='accuracy')
                
                # Also get predictions for confidence analysis
                cv_preds = cross_val_predict(ensemble, X, y, cv=skf)
                
                ensemble_results[name] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std(),
                    'scores': scores.tolist(),
                    'cv_predictions': cv_preds.tolist()
                }
                
                logger.info(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                ensemble_results[name] = {
                    'mean_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'scores': [0.0] * cv,
                    'cv_predictions': []
                }
        
        self.results['ensembles'] = ensemble_results
        return ensemble_results
    
    def optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters for best performing models"""
        logger.info("Optimizing hyperparameters...")
        
        # Select top 3 base models for optimization
        base_results = self.results.get('base_models', {})
        top_models = sorted(
            base_results.items(),
            key=lambda x: x[1]['mean_accuracy'],
            reverse=True
        )[:3]
        
        optimized_models = []
        
        for name, _ in top_models:
            if name == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 8, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                grid_search = RandomizedSearchCV(
                    RandomForestClassifier(random_state=self.random_state),
                    param_grid,
                    n_iter=20,
                    cv=3,
                    scoring='accuracy',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
            elif name == 'lr':
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
                
                grid_search = GridSearchCV(
                    LogisticRegression(max_iter=1000, random_state=self.random_state),
                    param_grid,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1
                )
            
            elif name == 'svm_rbf':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
                
                grid_search = GridSearchCV(
                    SVC(kernel='rbf', probability=True, random_state=self.random_state),
                    param_grid,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1
                )
            
            else:
                continue
            
            try:
                grid_search.fit(X, y)
                optimized_models.append((f'{name}_optimized', grid_search.best_estimator_))
                logger.info(f"Optimized {name}: {grid_search.best_score_:.3f}")
            except Exception as e:
                logger.warning(f"Failed to optimize {name}: {e}")
        
        # Add optimized models to base models
        self.base_models.extend(optimized_models)
        
        return optimized_models
    
    def confidence_weighted_predictions(self, X, y):
        """Create confidence-weighted ensemble predictions"""
        logger.info("Creating confidence-weighted predictions...")
        
        # Train models and get prediction confidence
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=self.random_state
        )
        
        model_confidences = {}
        model_predictions = {}
        
        for name, model in self.base_models:
            try:
                fitted_model = model.fit(X_train, y_train)
                
                if hasattr(fitted_model, 'predict_proba'):
                    proba = fitted_model.predict_proba(X_test)
                    confidence = np.max(proba, axis=1)
                    predictions = fitted_model.predict(X_test)
                else:
                    predictions = fitted_model.predict(X_test)
                    confidence = np.ones(len(predictions))  # Default confidence
                
                model_confidences[name] = confidence
                model_predictions[name] = predictions
                
            except Exception as e:
                logger.warning(f"Failed confidence analysis for {name}: {e}")
        
        # Weighted predictions based on confidence
        final_predictions = []
        
        for i in range(len(X_test)):
            weighted_votes = {}
            total_weight = 0
            
            for name in model_predictions:
                pred = model_predictions[name][i]
                confidence = model_confidences[name][i]
                
                if pred not in weighted_votes:
                    weighted_votes[pred] = 0
                
                weighted_votes[pred] += confidence
                total_weight += confidence
            
            # Normalize weights
            for pred in weighted_votes:
                weighted_votes[pred] /= total_weight
            
            # Select prediction with highest weighted vote
            final_pred = max(weighted_votes, key=weighted_votes.get)
            final_predictions.append(final_pred)
        
        confidence_accuracy = accuracy_score(y_test, final_predictions)
        
        self.results['confidence_weighted'] = {
            'accuracy': confidence_accuracy,
            'predictions': final_predictions,
            'true_labels': y_test.tolist()
        }
        
        logger.info(f"Confidence-weighted accuracy: {confidence_accuracy:.3f}")
        
        return confidence_accuracy
    
    def generate_feature_importance(self, X, y):
        """Generate ensemble feature importance"""
        logger.info("Generating feature importance...")
        
        # Get feature importance from tree-based models
        tree_models = ['rf', 'rf_deep', 'et', 'gb']
        if HAS_XGB:
            tree_models.append('xgb')
        if HAS_LGB:
            tree_models.append('lgb')
        
        feature_importance = np.zeros(X.shape[1])
        importance_count = 0
        
        for name, model in self.base_models:
            if name in tree_models:
                try:
                    fitted_model = model.fit(X, y)
                    if hasattr(fitted_model, 'feature_importances_'):
                        feature_importance += fitted_model.feature_importances_
                        importance_count += 1
                except:
                    continue
        
        # Average importance
        if importance_count > 0:
            feature_importance /= importance_count
        
        # Create feature importance dataframe
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        self.results['feature_importance'] = importance_df.to_dict('records')
        
        return importance_df
    
    def create_visualizations(self, save_dir='model_outputs'):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Advanced Ensemble Performance Analysis', fontsize=16, fontweight='bold')
        
        # Base models performance
        base_results = self.results.get('base_models', {})
        if base_results:
            names = list(base_results.keys())
            means = [base_results[name]['mean_accuracy'] for name in names]
            stds = [base_results[name]['std_accuracy'] for name in names]
            
            axes[0, 0].barh(names, means, xerr=stds, capsize=5)
            axes[0, 0].set_xlabel('Accuracy')
            axes[0, 0].set_title('Base Models Performance')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Ensemble performance
        ensemble_results = self.results.get('ensembles', {})
        if ensemble_results:
            ens_names = list(ensemble_results.keys())
            ens_means = [ensemble_results[name]['mean_accuracy'] for name in ens_names]
            ens_stds = [ensemble_results[name]['std_accuracy'] for name in ens_names]
            
            bars = axes[0, 1].bar(range(len(ens_names)), ens_means, 
                                yerr=ens_stds, capsize=5, alpha=0.7)
            axes[0, 1].set_xlabel('Ensemble Methods')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Ensemble Performance Comparison')
            axes[0, 1].set_xticks(range(len(ens_names)))
            axes[0, 1].set_xticklabels(ens_names, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Color best performing ensemble
            if ens_means:
                best_idx = np.argmax(ens_means)
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('black')
                bars[best_idx].set_linewidth(2)
        
        # Feature importance
        feature_importance = self.results.get('feature_importance', [])
        if feature_importance:
            top_features = feature_importance[:15]  # Top 15 features
            feature_names = [f['feature'] for f in top_features]
            importances = [f['importance'] for f in top_features]
            
            axes[1, 0].barh(feature_names, importances)
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top 15 Feature Importance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Performance distribution
        all_scores = []
        all_labels = []
        
        for name, results in base_results.items():
            all_scores.extend(results['scores'])
            all_labels.extend([f'Base: {name}'] * len(results['scores']))
        
        for name, results in ensemble_results.items():
            all_scores.extend(results['scores'])
            all_labels.extend([f'Ens: {name}'] * len(results['scores']))
        
        if all_scores:
            df_scores = pd.DataFrame({'Score': all_scores, 'Model': all_labels})
            
            # Group by model type
            df_scores['Type'] = df_scores['Model'].apply(lambda x: x.split(':')[0])
            
            sns.boxplot(data=df_scores, x='Type', y='Score', ax=axes[1, 1])
            axes[1, 1].set_title('Score Distribution by Model Type')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/advanced_ensemble_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed ensemble comparison
        if ensemble_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create detailed comparison
            ens_names = list(ensemble_results.keys())
            ens_means = [ensemble_results[name]['mean_accuracy'] for name in ens_names]
            ens_stds = [ensemble_results[name]['std_accuracy'] for name in ens_names]
            
            # Sort by performance
            sorted_indices = np.argsort(ens_means)[::-1]
            sorted_names = [ens_names[i] for i in sorted_indices]
            sorted_means = [ens_means[i] for i in sorted_indices]
            sorted_stds = [ens_stds[i] for i in sorted_indices]
            
            bars = ax.barh(range(len(sorted_names)), sorted_means, 
                          xerr=sorted_stds, capsize=5, alpha=0.8)
            
            # Color code by performance
            for i, bar in enumerate(bars):
                if sorted_means[i] >= 0.90:
                    bar.set_color('gold')
                elif sorted_means[i] >= 0.85:
                    bar.set_color('silver')
                else:
                    bar.set_color('lightcoral')
            
            ax.set_yticks(range(len(sorted_names)))
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Accuracy')
            ax.set_title('Ensemble Methods Ranked by Performance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add performance labels
            for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):
                ax.text(mean + 0.01, i, f'{mean:.3f} ± {std:.3f}', 
                       va='center', fontweight='bold')
            
            # Add target line at 90%
            ax.axvline(x=0.90, color='red', linestyle='--', alpha=0.7, 
                      label='90% Target')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/ensemble_ranking.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {save_dir}/")
    
    def save_results(self, save_dir='model_outputs'):
        """Save comprehensive results"""
        logger.info("Saving results...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'{save_dir}/advanced_ensemble_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary report
        report_file = f'{save_dir}/ensemble_summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("MPEG-G Advanced Ensemble Pipeline Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random Seed: {self.random_state}\n\n")
            
            # Base models summary
            f.write("BASE MODELS PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            
            base_results = self.results.get('base_models', {})
            if base_results:
                sorted_base = sorted(
                    base_results.items(),
                    key=lambda x: x[1]['mean_accuracy'],
                    reverse=True
                )
                
                for name, results in sorted_base:
                    f.write(f"{name:20}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}\n")
            
            f.write("\n")
            
            # Ensemble results summary
            f.write("ENSEMBLE METHODS PERFORMANCE:\n")
            f.write("-" * 35 + "\n")
            
            ensemble_results = self.results.get('ensembles', {})
            if ensemble_results:
                sorted_ensemble = sorted(
                    ensemble_results.items(),
                    key=lambda x: x[1]['mean_accuracy'],
                    reverse=True
                )
                
                best_accuracy = 0
                best_method = None
                
                for name, results in sorted_ensemble:
                    accuracy = results['mean_accuracy']
                    std = results['std_accuracy']
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_method = name
                    
                    status = ""
                    if accuracy >= 0.90:
                        status = " ⭐ TARGET ACHIEVED"
                    elif accuracy >= 0.85:
                        status = " ✓ IMPROVED"
                    
                    f.write(f"{name:25}: {accuracy:.3f} ± {std:.3f}{status}\n")
                
                f.write("\n")
                f.write(f"BEST PERFORMING METHOD: {best_method}\n")
                f.write(f"BEST ACCURACY: {best_accuracy:.3f}\n")
                
                improvement = ((best_accuracy - 0.85) / 0.85) * 100
                f.write(f"IMPROVEMENT OVER BASELINE: +{improvement:.1f}%\n")
            
            # Confidence weighted results
            conf_results = self.results.get('confidence_weighted', {})
            if conf_results:
                f.write(f"\nCONFIDENCE-WEIGHTED ACCURACY: {conf_results['accuracy']:.3f}\n")
            
            # Feature importance summary
            feature_importance = self.results.get('feature_importance', [])
            if feature_importance:
                f.write("\nTOP 10 MOST IMPORTANT FEATURES:\n")
                f.write("-" * 35 + "\n")
                
                for i, feature in enumerate(feature_importance[:10], 1):
                    f.write(f"{i:2}. {feature['feature']:30}: {feature['importance']:.4f}\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS FOR TRACK 1 SUBMISSION:\n")
            f.write("-" * 45 + "\n")
            
            if ensemble_results:
                top_3_methods = sorted(
                    ensemble_results.items(),
                    key=lambda x: x[1]['mean_accuracy'],
                    reverse=True
                )[:3]
                
                f.write("1. PRIMARY RECOMMENDATION:\n")
                best_name, best_results = top_3_methods[0]
                f.write(f"   Use {best_name} (Accuracy: {best_results['mean_accuracy']:.3f})\n")
                f.write(f"   Variance: ±{best_results['std_accuracy']:.3f}\n\n")
                
                f.write("2. BACKUP OPTIONS:\n")
                for i, (name, results) in enumerate(top_3_methods[1:], 2):
                    f.write(f"   {i}. {name}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}\n")
                
                f.write("\n3. CONFIDENCE ESTIMATES:\n")
                f.write("   High confidence predictions when multiple methods agree\n")
                f.write("   Monitor prediction variance across ensemble members\n")
        
        logger.info(f"Results saved to {save_dir}/")
        return results_file, report_file
    
    def run_complete_pipeline(self, X, y):
        """Run the complete advanced ensemble pipeline"""
        logger.info("Starting Advanced Ensemble Pipeline for MPEG-G Challenge")
        logger.info(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        # 1. Evaluate base models
        self.evaluate_base_models(X, y, cv=5)
        
        # 2. Optimize hyperparameters for top models
        self.optimize_hyperparameters(X, y)
        
        # 3. Re-evaluate with optimized models
        self.evaluate_base_models(X, y, cv=5)
        
        # 4. Create ensemble methods
        self.create_voting_ensembles(X, y)
        self.create_stacking_ensemble(X, y)
        self.create_bagging_ensembles(X, y)
        self.create_dynamic_ensemble(X, y)
        
        # 5. Diversity-based pruning
        self.diversity_based_pruning(X, y)
        
        # 6. Evaluate all ensembles
        self.evaluate_ensembles(X, y, cv=5)
        
        # 7. Confidence-weighted predictions
        self.confidence_weighted_predictions(X, y)
        
        # 8. Feature importance analysis
        self.generate_feature_importance(X, y)
        
        # 9. Create visualizations
        self.create_visualizations()
        
        # 10. Save results
        self.save_results()
        
        # Summary
        ensemble_results = self.results.get('ensembles', {})
        if ensemble_results:
            best_method = max(ensemble_results.items(), 
                            key=lambda x: x[1]['mean_accuracy'])
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETE - SUMMARY RESULTS")
            logger.info("="*60)
            logger.info(f"Best Method: {best_method[0]}")
            logger.info(f"Best Accuracy: {best_method[1]['mean_accuracy']:.3f} ± {best_method[1]['std_accuracy']:.3f}")
            
            improvement = ((best_method[1]['mean_accuracy'] - 0.85) / 0.85) * 100
            logger.info(f"Improvement over 85% baseline: +{improvement:.1f}%")
            
            if best_method[1]['mean_accuracy'] >= 0.90:
                logger.info("🎯 TARGET ACHIEVED: 90%+ accuracy reached!")
            elif best_method[1]['mean_accuracy'] >= 0.87:
                logger.info("✅ SIGNIFICANT IMPROVEMENT achieved")
            else:
                logger.info("📈 Modest improvement, consider additional techniques")
            
            logger.info("="*60)
        
        return self.results


def main():
    """Main execution function"""
    
    # Load enhanced features data
    logger.info("Loading enhanced features data...")
    
    data_dir = "/Users/sayantanpal100/Desktop/MPEG-G_ Decoding the Dialogue/enhanced_features"
    
    try:
        # Load features and metadata
        features_df = pd.read_csv(f"{data_dir}/enhanced_features_final.csv", index_col=0)
        metadata_df = pd.read_csv(f"{data_dir}/enhanced_metadata_final.csv", index_col=0)
        
        logger.info(f"Loaded features: {features_df.shape}")
        logger.info(f"Loaded metadata: {metadata_df.shape}")
        
        # Prepare data
        X = features_df.values
        y = metadata_df['symptom'].values
        
        # Encode target labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        logger.info(f"Classes: {le.classes_}")
        logger.info(f"Class distribution: {np.bincount(y_encoded)}")
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize and run pipeline
        pipeline = AdvancedEnsemblePipeline(random_state=42)
        results = pipeline.run_complete_pipeline(X_scaled, y_encoded)
        
        logger.info("Advanced Ensemble Pipeline completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()