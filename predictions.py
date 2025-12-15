"""
Tuned Prediction Heads for ESM2 Embeddings
===========================================
Optimized for: ~1200 samples, 1280 features (high-dimensional, small-sample)

Key adjustments:
- Stronger regularization (more features than samples)
- Shallower trees (prevent overfitting)
- Smaller neural networks
- PCA option for dimensionality reduction
- Realistic time estimates


To compare raw embeddings vs PCA performance:
python predictions.py --compare

To run only on PCA:
python predictions.py

To run on both:
python predictions.py --full
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import warnings
import logging
import time
from dataclasses import dataclass

from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_predict, cross_val_score,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)

# Optional boosting libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Column names
COL_GENE = 'Well ID'
COL_AVG = 'Average (%Myh6+ cells)'
COL_STDEV = 'StDev (%Myh6+ cells)'
COL_PVAL = 'P-Value (%Myh6+ cells)'
COL_CELLS = 'Total Cells (CellScoring)'
COL_TRANSFORMS = 'Transforms'


@dataclass
class ModelResult:
    """Container for model results."""
    model_name: str
    target: str
    task_type: str
    cv_score_mean: float
    cv_score_std: float
    best_params: Optional[dict]
    predictions: np.ndarray
    true_values: np.ndarray
    probabilities: Optional[np.ndarray] = None
    train_time: float = 0.0
    
    # Metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None


class TunedPredictor:
    """
    Prediction heads tuned for high-dimensional, small-sample data.
    
    Data profile: ~1200 samples, 1280 features
    """
    
    def __init__(
        self,
        embedding_path: str,
        target_csv: str,
        gene_col: str = COL_GENE,
        use_pca: bool = True,  # Recommended for this data!
        pca_components: int = 256,  # Reduce 1280 -> 256
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.embedding_path = Path(embedding_path)
        self.target_csv = Path(target_csv)
        self.gene_col = gene_col
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.X: Optional[np.ndarray] = None
        self.targets: Optional[pd.DataFrame] = None
        self.genes: Optional[np.ndarray] = None
        
        self.results: dict[str, dict[str, ModelResult]] = {}
        self.best_models: dict[str, str] = {}
    
    def _get_classification_models(self) -> dict:
        """
        Classification models tuned for 1200 samples, 1280 features.
        
        Estimated times per model (with n_iter=20, cv=5):
        - LogisticRegression: ~1-2 min
        - RandomForest: ~3-5 min
        - GradientBoosting: ~5-8 min
        - ExtraTrees: ~2-4 min
        - SVM: ~5-10 min (can be slow)
        - MLP: ~3-5 min
        - XGBoost: ~2-3 min
        - LightGBM: ~1-2 min
        """
        models = {
            # Strong L1/L2 regularization for high-dim
            'LogisticRegression': {
                'model': LogisticRegression(
                    class_weight='balanced',
                    max_iter=2000,
                    random_state=self.random_state,
                    solver='saga',  # Supports L1
                ),
                'params': {
                    'model__C': [0.001, 0.01, 0.1, 1.0],  # Stronger regularization
                    'model__penalty': ['l1', 'l2', 'elasticnet'],
                    'model__l1_ratio': [0.5],  # For elasticnet
                },
                'est_time': '1-2 min'
            },
            
            # Shallow trees, fewer estimators
            'RandomForest': {
                'model': RandomForestClassifier(
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_features='sqrt',  # Important for high-dim
                ),
                'params': {
                    'model__n_estimators': [100, 150],  # Reduced from 300
                    'model__max_depth': [3, 5, 7],  # Shallow! Was 5-15
                    'model__min_samples_leaf': [5, 10, 20],  # More conservative
                    'model__min_samples_split': [10, 20],
                },
                'est_time': '3-5 min'
            },
            
            # Conservative boosting
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    random_state=self.random_state,
                    max_features='sqrt',
                ),
                'params': {
                    'model__n_estimators': [50, 100],  # Fewer trees
                    'model__max_depth': [2, 3, 4],  # Very shallow
                    'model__learning_rate': [0.05, 0.1],  # Lower LR
                    'model__min_samples_leaf': [10, 20],
                    'model__subsample': [0.8],  # Regularization
                },
                'est_time': '5-8 min'
            },
            
            # ExtraTrees - often good for high-dim
            'ExtraTrees': {
                'model': ExtraTreesClassifier(
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_features='sqrt',
                ),
                'params': {
                    'model__n_estimators': [100, 150],
                    'model__max_depth': [5, 7, 10],
                    'model__min_samples_leaf': [5, 10],
                },
                'est_time': '2-4 min'
            },
            
            # SVM with RBF - good for high-dim but slow
            'SVM': {
                'model': SVC(
                    class_weight='balanced',
                    probability=True,
                    random_state=self.random_state,
                    cache_size=1000,  # Speed up
                ),
                'params': {
                    'model__C': [0.1, 1.0, 10.0],
                    'model__kernel': ['rbf'],
                    'model__gamma': ['scale', 0.001, 0.01],
                },
                'est_time': '5-10 min'
            },
            
            # Small MLP - avoid overfitting
            'MLP': {
                'model': MLPClassifier(
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=self.random_state,
                ),
                'params': {
                    # Smaller networks for small data!
                    'model__hidden_layer_sizes': [(64,), (128,), (64, 32)],
                    'model__alpha': [0.01, 0.1, 1.0],  # Strong L2!
                    'model__learning_rate_init': [0.001],
                    'model__batch_size': [32, 64],
                },
                'est_time': '3-5 min'
            },
        }
        
        if HAS_XGB:
            models['XGBoost'] = {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    eval_metric='logloss',
                    use_label_encoder=False,
                ),
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [2, 3, 4],  # Shallow
                    'model__learning_rate': [0.05, 0.1],
                    'model__subsample': [0.8],
                    'model__colsample_bytree': [0.5, 0.8],  # Feature sampling
                    'model__reg_alpha': [0.1, 1.0],  # L1
                    'model__reg_lambda': [1.0, 10.0],  # L2
                },
                'est_time': '2-3 min'
            }
        
        if HAS_LGBM:
            models['LightGBM'] = {
                'model': LGBMClassifier(
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                ),
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [3, 5],
                    'model__learning_rate': [0.05, 0.1],
                    'model__num_leaves': [15, 31],  # Conservative
                    'model__reg_alpha': [0.1, 1.0],
                    'model__reg_lambda': [1.0, 10.0],
                    'model__feature_fraction': [0.5, 0.8],
                },
                'est_time': '1-2 min'
            }
        
        return models
    
    def _get_regression_models(self) -> dict:
        """Regression models tuned for high-dimensional data."""
        models = {
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'model__alpha': [1, 10, 100, 1000],  # Strong regularization
                },
                'est_time': '<1 min'
            },
            
            'ElasticNet': {
                'model': ElasticNet(max_iter=5000),
                'params': {
                    'model__alpha': [0.1, 1, 10],
                    'model__l1_ratio': [0.2, 0.5, 0.8],
                },
                'est_time': '<1 min'
            },
            
            'RandomForest': {
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_features='sqrt',
                ),
                'params': {
                    'model__n_estimators': [100, 150],
                    'model__max_depth': [3, 5, 7],
                    'model__min_samples_leaf': [5, 10, 20],
                },
                'est_time': '3-5 min'
            },
            
            'GradientBoosting': {
                'model': GradientBoostingRegressor(
                    random_state=self.random_state,
                    max_features='sqrt',
                ),
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [2, 3, 4],
                    'model__learning_rate': [0.05, 0.1],
                    'model__min_samples_leaf': [10, 20],
                },
                'est_time': '5-8 min'
            },
            
            'SVR': {
                'model': SVR(),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf'],
                    'model__gamma': ['scale', 0.001],
                },
                'est_time': '5-10 min'
            },
            
            'MLP': {
                'model': MLPRegressor(
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=self.random_state,
                ),
                'params': {
                    'model__hidden_layer_sizes': [(64,), (128,), (64, 32)],
                    'model__alpha': [0.01, 0.1, 1.0],
                    'model__learning_rate_init': [0.001],
                },
                'est_time': '3-5 min'
            },
        }
        
        if HAS_XGB:
            models['XGBoost'] = {
                'model': XGBRegressor(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [2, 3, 4],
                    'model__learning_rate': [0.05, 0.1],
                    'model__reg_alpha': [0.1, 1.0],
                    'model__reg_lambda': [1.0, 10.0],
                },
                'est_time': '2-3 min'
            }
        
        if HAS_LGBM:
            models['LightGBM'] = {
                'model': LGBMRegressor(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                ),
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [3, 5],
                    'model__learning_rate': [0.05, 0.1],
                    'model__reg_alpha': [0.1, 1.0],
                    'model__reg_lambda': [1.0, 10.0],
                },
                'est_time': '1-2 min'
            }
        
        return models
    
    def load_data(self):
        """Load and preprocess data."""
        logger.info(f"Loading embeddings from {self.embedding_path}")
        data = np.load(self.embedding_path, allow_pickle=True)
        emb_genes = list(data['genes'])
        embeddings = data['embeddings']
        
        logger.info(f"Loading targets from {self.target_csv}")
        target_df = pd.read_csv(self.target_csv)
        
        # Analyze controls
        control_mask = target_df[self.gene_col].str.lower() == 'control'
        n_controls = control_mask.sum()
        
        if n_controls > 0:
            control_avg = target_df.loc[control_mask, COL_AVG]
            baseline = control_avg.mean()
            
            logger.info(f"\nControl wells: {n_controls}")
            logger.info(f"  Baseline (mean): {baseline:.2f}%")
            logger.info(f"  Range: [{control_avg.min():.2f}%, {control_avg.max():.2f}%]")
            
            # Create binary target
            is_hit = (target_df[COL_PVAL] < 0.05) & (target_df[COL_AVG] > baseline)
            target_df[COL_TRANSFORMS] = is_hit.astype(float)
            target_df.loc[control_mask, COL_TRANSFORMS] = np.nan
            
            # Exclude controls
            target_df = target_df[~control_mask].copy()
        
        # Align with embeddings
        gene_to_idx = {g: i for i, g in enumerate(emb_genes)}
        valid_mask = target_df[self.gene_col].isin(gene_to_idx.keys())
        target_df = target_df[valid_mask].copy()
        
        self.genes = target_df[self.gene_col].values
        self.X = np.array([embeddings[gene_to_idx[g]] for g in self.genes])
        self.targets = target_df[[COL_TRANSFORMS, COL_AVG, COL_STDEV, COL_PVAL, COL_CELLS]].copy()
        
        # Class distribution
        y_class = self.targets[COL_TRANSFORMS].dropna()
        n_hits = int(y_class.sum())
        n_total = len(y_class)
        
        logger.info(f"\n{'='*60}")
        logger.info("DATA SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Samples:  {self.X.shape[0]}")
        logger.info(f"Features: {self.X.shape[1]}")
        logger.info(f"Ratio:    {self.X.shape[0]/self.X.shape[1]:.2f} samples/feature")
        logger.info(f"\nClass distribution (Transforms):")
        logger.info(f"  Hits (1):     {n_hits} ({100*n_hits/n_total:.1f}%)")
        logger.info(f"  Non-hits (0): {n_total-n_hits} ({100*(n_total-n_hits)/n_total:.1f}%)")
        
        if self.use_pca:
            logger.info(f"\nPCA: Reducing {self.X.shape[1]} -> {self.pca_components} dimensions")
        
        return self.X, self.targets, self.genes
    
    def _build_pipeline(self, model) -> Pipeline:
        """Build preprocessing pipeline with optional PCA."""
        steps = [('scaler', RobustScaler())]
        
        if self.use_pca:
            steps.append(('pca', PCA(
                n_components=self.pca_components,
                random_state=self.random_state
            )))
        
        steps.append(('model', model))
        return Pipeline(steps)
    
    def estimate_total_time(self, targets: list[str], n_iter: int = 20) -> str:
        """Estimate total training time."""
        # Base times per model (minutes) with PCA
        base_times = {
            'LogisticRegression': 1.5,
            'RandomForest': 4,
            'GradientBoosting': 6,
            'ExtraTrees': 3,
            'SVM': 7,
            'MLP': 4,
            'XGBoost': 2.5,
            'LightGBM': 1.5,
        }
        
        n_models = len(base_times)
        if HAS_XGB:
            n_models += 0  # Already counted
        else:
            base_times.pop('XGBoost', None)
        if not HAS_LGBM:
            base_times.pop('LightGBM', None)
        
        # Scale by n_iter (base assumes n_iter=20)
        scale = n_iter / 20
        
        total_minutes = sum(base_times.values()) * scale * len(targets)
        
        if total_minutes < 60:
            return f"~{int(total_minutes)} minutes"
        else:
            return f"~{total_minutes/60:.1f} hours"
    
    def train_target(
        self,
        target_col: str,
        cv_folds: int = 5,
        n_iter: int = 20,
    ) -> dict[str, ModelResult]:
        """Train all models for a single target."""
        if self.X is None:
            self.load_data()
        
        y = self.targets[target_col].values
        valid_mask = ~pd.isna(y)
        X = self.X[valid_mask]
        y = y[valid_mask]
        
        # Task type
        unique_vals = np.unique(y)
        is_classification = len(unique_vals) <= 10
        task_type = 'classification' if is_classification else 'regression'
        
        if is_classification:
            y = y.astype(int)
            models = self._get_classification_models()
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'f1'
        else:
            models = self._get_regression_models()
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'r2'
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TARGET: {target_col} ({task_type.upper()})")
        logger.info(f"{'='*70}")
        logger.info(f"CV: {cv_folds}-fold, Hyperparam iterations: {n_iter}")
        
        results = {}
        total_start = time.time()
        
        for model_name, config in models.items():
            logger.info(f"\n--- {model_name} (est. {config['est_time']}) ---")
            start_time = time.time()
            
            try:
                pipeline = self._build_pipeline(config['model'])
                
                # Count total parameter combinations
                n_combinations = np.prod([len(v) for v in config['params'].values()])
                actual_iter = min(n_iter, n_combinations)
                
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=config['params'],
                    n_iter=actual_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    refit=True,
                    verbose=0,
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X, y)
                
                # Get CV predictions
                y_pred = cross_val_predict(search.best_estimator_, X, y, cv=cv)
                
                y_prob = None
                if is_classification:
                    try:
                        y_prob = cross_val_predict(
                            search.best_estimator_, X, y, cv=cv, method='predict_proba'
                        )
                        if y_prob.ndim == 2:
                            y_prob = y_prob[:, 1]
                    except:
                        pass
                
                elapsed = time.time() - start_time
                
                # Create result
                result = ModelResult(
                    model_name=model_name,
                    target=target_col,
                    task_type=task_type,
                    cv_score_mean=search.best_score_,
                    cv_score_std=search.cv_results_['std_test_score'][search.best_index_],
                    best_params={k.replace('model__', ''): v for k, v in search.best_params_.items()},
                    predictions=y_pred,
                    true_values=y,
                    probabilities=y_prob,
                    train_time=elapsed,
                )
                
                # Metrics
                if is_classification:
                    result.accuracy = accuracy_score(y, y_pred)
                    result.precision = precision_score(y, y_pred, zero_division=0)
                    result.recall = recall_score(y, y_pred, zero_division=0)
                    result.f1 = f1_score(y, y_pred, zero_division=0)
                    if y_prob is not None:
                        result.auc_roc = roc_auc_score(y, y_prob)
                        result.auc_pr = average_precision_score(y, y_prob)
                    
                    auc_str = f"{result.auc_roc:.3f}" if result.auc_roc else "N/A"
                    logger.info(f"  F1={result.f1:.3f} | AUC-ROC={auc_str} | "
                               f"Prec={result.precision:.3f} | Rec={result.recall:.3f}")
                else:
                    result.r2 = r2_score(y, y_pred)
                    result.rmse = np.sqrt(mean_squared_error(y, y_pred))
                    result.mae = mean_absolute_error(y, y_pred)
                    logger.info(f"  R²={result.r2:.3f} | RMSE={result.rmse:.3f} | MAE={result.mae:.3f}")
                
                logger.info(f"  Time: {elapsed:.1f}s | Best: {result.best_params}")
                
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"  Failed: {e}")
        
        total_elapsed = time.time() - total_start
        logger.info(f"\nTotal time for {target_col}: {total_elapsed/60:.1f} minutes")
        
        # Find best
        if results:
            if is_classification:
                best_name = max(results, key=lambda k: results[k].f1 or 0)
                best_score = results[best_name].f1
            else:
                best_name = max(results, key=lambda k: results[k].r2 or -999)
                best_score = results[best_name].r2
            
            self.best_models[target_col] = best_name
            logger.info(f"\n★ BEST: {best_name} (score={best_score:.3f})")
        
        self.results[target_col] = results
        return results
    
    def train_all(
        self,
        targets: list[str] = None,
        cv_folds: int = 5,
        n_iter: int = 20,
    ):
        """Train models for all targets."""
        if self.X is None:
            self.load_data()
        
        if targets is None:
            targets = [COL_TRANSFORMS, COL_AVG]
        
        est_time = self.estimate_total_time(targets, n_iter)
        logger.info(f"\nEstimated total time: {est_time}")
        logger.info("="*70)
        
        for target in targets:
            if target in self.targets.columns:
                self.train_target(target, cv_folds=cv_folds, n_iter=n_iter)
        
        return self.results
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        rows = []
        for target, models in self.results.items():
            for name, res in models.items():
                row = {
                    'Target': target,
                    'Model': name,
                    'Task': res.task_type,
                    'Time (s)': f"{res.train_time:.1f}",
                }
                if res.task_type == 'classification':
                    row.update({
                        'F1': f"{res.f1:.3f}" if res.f1 else "N/A",
                        'AUC-ROC': f"{res.auc_roc:.3f}" if res.auc_roc else "N/A",
                        'AUC-PR': f"{res.auc_pr:.3f}" if res.auc_pr else "N/A",
                        'Accuracy': f"{res.accuracy:.3f}" if res.accuracy else "N/A",
                    })
                else:
                    row.update({
                        'R²': f"{res.r2:.3f}" if res.r2 else "N/A",
                        'RMSE': f"{res.rmse:.3f}" if res.rmse else "N/A",
                        'MAE': f"{res.mae:.3f}" if res.mae else "N/A",
                    })
                rows.append(row)
        return pd.DataFrame(rows)
    
    def plot_comparison(self, target_col: str, output_path: str = None):
        """Plot model comparison."""
        if target_col not in self.results:
            return
        
        results = self.results[target_col]
        task_type = list(results.values())[0].task_type
        
        models = list(results.keys())
        
        if task_type == 'classification':
            scores = [results[m].f1 or 0 for m in models]
            metric = 'F1 Score'
        else:
            scores = [results[m].r2 or 0 for m in models]
            metric = 'R² Score'
        
        # Sort
        sorted_idx = np.argsort(scores)
        sorted_models = [models[i] for i in sorted_idx]
        sorted_scores = [scores[i] for i in sorted_idx]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))
        
        bars = ax.barh(sorted_models, sorted_scores, color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f'{target_col}: Model Comparison')
        ax.set_xlim(0, max(1, max(sorted_scores) * 1.1))
        
        # Add value labels
        for bar, score in zip(bars, sorted_scores):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, target_col: str, model_name: str = None, output_path: str = None):
        """Plot confusion matrix."""
        if target_col not in self.results:
            return
        
        if model_name is None:
            model_name = self.best_models.get(target_col)
        
        res = self.results[target_col][model_name]
        cm = confusion_matrix(res.true_values, res.predictions)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max()/2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                       color=color, fontsize=16)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nNon-hit', 'Predicted\nHit'])
        ax.set_yticklabels(['Actual\nNon-hit', 'Actual\nHit'])
        ax.set_title(f'{model_name}\nF1={res.f1:.3f}, AUC={res.auc_roc:.3f if res.auc_roc else 0:.3f}')
        
        plt.colorbar(im)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, target_col: str, model_name: str = None, output_path: str = None):
        """Plot predicted vs actual for regression."""
        if target_col not in self.results:
            return
        
        if model_name is None:
            model_name = self.best_models.get(target_col)
        
        res = self.results[target_col][model_name]
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(res.true_values, res.predictions, alpha=0.5, edgecolors='k', linewidth=0.3)
        
        lims = [
            min(res.true_values.min(), res.predictions.min()),
            max(res.true_values.max(), res.predictions.max())
        ]
        ax.plot(lims, lims, 'r--', alpha=0.8, label='Perfect prediction')
        
        ax.set_xlabel(f'Actual {target_col}')
        ax.set_ylabel(f'Predicted {target_col}')
        ax.set_title(f'{model_name}\nR²={res.r2:.3f}, RMSE={res.rmse:.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def run_pipeline(
    embedding_path: str = 'tf_embeddings_output/embeddings_1280.npz',
    target_csv: str = 'tf.csv',
    output_dir: str = 'prediction_results',
    cv_folds: int = 5,
    n_iter: int = 20,
    use_pca: bool = True,
    pca_components: int = 256,
    targets: list[str] = None,
):
    """
    Run the full tuned prediction pipeline.
    
    Time estimates (with PCA, n_iter=20):
    - Per target: ~25-35 minutes
    - Two targets: ~50-70 minutes
    
    For faster results, reduce n_iter to 10 (~half the time).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    predictor = TunedPredictor(
        embedding_path=embedding_path,
        target_csv=target_csv,
        use_pca=use_pca,
        pca_components=pca_components,
    )
    
    # Load data
    predictor.load_data()
    
    # Train
    if targets is None:
        targets = [COL_TRANSFORMS, COL_AVG]
    
    predictor.train_all(targets=targets, cv_folds=cv_folds, n_iter=n_iter)
    
    # Summary
    summary = predictor.get_summary()
    summary.to_csv(output_dir / 'results_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(summary.to_string(index=False))
    
    # Plots
    for target in targets:
        if target not in predictor.results:
            continue
        
        safe_name = target.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '')
        
        predictor.plot_comparison(target, output_dir / f'{safe_name}_comparison.png')
        
        task_type = list(predictor.results[target].values())[0].task_type
        if task_type == 'classification':
            predictor.plot_confusion_matrix(target, output_path=output_dir / f'{safe_name}_confusion.png')
        else:
            predictor.plot_predictions(target, output_path=output_dir / f'{safe_name}_scatter.png')
    
    print(f"\nResults saved to: {output_dir}/")
    
    return predictor


def run_full_comparison(
    embedding_path: str = 'tf_embeddings_output/embeddings_1280.npz',
    target_csv: str = 'tf.csv',
    output_dir: str = 'prediction_results',
    cv_folds: int = 5,
    n_iter: int = 20,
    pca_components: int = 256,
    targets: list[str] = None,
):
    """
    Run FULL pipeline on BOTH raw embeddings and PCA.
    Compares all models on both representations.
    
    Time: ~80-120 min (runs pipeline twice)
    
    Returns dict with results from both configurations.
    """
    if targets is None:
        targets = [COL_TRANSFORMS, COL_AVG]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # ===== RUN 1: Raw Embeddings =====
    print("\n" + "="*70)
    print("CONFIGURATION 1: RAW EMBEDDINGS")
    print("="*70)
    
    predictor_raw = TunedPredictor(
        embedding_path=embedding_path,
        target_csv=target_csv,
        use_pca=False,
    )
    predictor_raw.load_data()
    predictor_raw.train_all(targets=targets, cv_folds=cv_folds, n_iter=n_iter)
    
    all_results['raw'] = {
        'predictor': predictor_raw,
        'summary': predictor_raw.get_summary(),
    }
    all_results['raw']['summary']['Config'] = 'Raw'
    
    # ===== RUN 2: PCA =====
    print("\n" + "="*70)
    print(f"CONFIGURATION 2: PCA ({pca_components} dims)")
    print("="*70)
    
    predictor_pca = TunedPredictor(
        embedding_path=embedding_path,
        target_csv=target_csv,
        use_pca=True,
        pca_components=pca_components,
    )
    predictor_pca.load_data()
    predictor_pca.train_all(targets=targets, cv_folds=cv_folds, n_iter=n_iter)
    
    all_results['pca'] = {
        'predictor': predictor_pca,
        'summary': predictor_pca.get_summary(),
    }
    all_results['pca']['summary']['Config'] = f'PCA-{pca_components}'
    
    # ===== COMBINED SUMMARY =====
    combined = pd.concat([
        all_results['raw']['summary'],
        all_results['pca']['summary']
    ], ignore_index=True)
    
    combined.to_csv(output_dir / 'full_comparison_results.csv', index=False)
    
    # Print comparison
    print("\n" + "="*70)
    print("FULL COMPARISON: RAW vs PCA")
    print("="*70)
    
    for target in targets:
        if target not in predictor_raw.results:
            continue
            
        task_type = list(predictor_raw.results[target].values())[0].task_type
        
        print(f"\n{target} ({task_type}):")
        print("-" * 60)
        
        if task_type == 'classification':
            # Get best from each config
            raw_results = predictor_raw.results[target]
            pca_results = predictor_pca.results[target]
            
            raw_best_name = max(raw_results, key=lambda k: raw_results[k].f1 or 0)
            pca_best_name = max(pca_results, key=lambda k: pca_results[k].f1 or 0)
            
            raw_best = raw_results[raw_best_name]
            pca_best = pca_results[pca_best_name]
            
            print(f"  {'Config':<15} | {'Best Model':<20} | {'F1':>6} | {'AUC-ROC':>7}")
            print(f"  {'-'*15}-+-{'-'*20}-+-{'-'*6}-+-{'-'*7}")
            print(f"  {'Raw (1280)':<15} | {raw_best_name:<20} | {raw_best.f1:.3f} | {raw_best.auc_roc:.3f if raw_best.auc_roc else 'N/A':>7}")
            print(f"  {f'PCA ({pca_components})':<15} | {pca_best_name:<20} | {pca_best.f1:.3f} | {pca_best.auc_roc:.3f if pca_best.auc_roc else 'N/A':>7}")
            
            # Winner
            if (pca_best.f1 or 0) > (raw_best.f1 or 0):
                winner = f"PCA ({pca_components})"
                winner_model = pca_best_name
                winner_score = pca_best.f1
            else:
                winner = "Raw (1280)"
                winner_model = raw_best_name
                winner_score = raw_best.f1
            
            print(f"\n  ★ WINNER: {winner} with {winner_model} (F1={winner_score:.3f})")
            
        else:
            raw_results = predictor_raw.results[target]
            pca_results = predictor_pca.results[target]
            
            raw_best_name = max(raw_results, key=lambda k: raw_results[k].r2 or -999)
            pca_best_name = max(pca_results, key=lambda k: pca_results[k].r2 or -999)
            
            raw_best = raw_results[raw_best_name]
            pca_best = pca_results[pca_best_name]
            
            print(f"  {'Config':<15} | {'Best Model':<20} | {'R²':>6} | {'RMSE':>7}")
            print(f"  {'-'*15}-+-{'-'*20}-+-{'-'*6}-+-{'-'*7}")
            print(f"  {'Raw (1280)':<15} | {raw_best_name:<20} | {raw_best.r2:.3f} | {raw_best.rmse:.3f}")
            print(f"  {f'PCA ({pca_components})':<15} | {pca_best_name:<20} | {pca_best.r2:.3f} | {pca_best.rmse:.3f}")
            
            if (pca_best.r2 or -999) > (raw_best.r2 or -999):
                winner = f"PCA ({pca_components})"
                winner_model = pca_best_name
                winner_score = pca_best.r2
            else:
                winner = "Raw (1280)"
                winner_model = raw_best_name
                winner_score = raw_best.r2
            
            print(f"\n  ★ WINNER: {winner} with {winner_model} (R²={winner_score:.3f})")
    
    # Plot comparison
    _plot_full_comparison(predictor_raw, predictor_pca, targets, pca_components, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}")
    
    return all_results


def _plot_full_comparison(predictor_raw, predictor_pca, targets, pca_dim, output_dir):
    """Plot side-by-side comparison of raw vs PCA."""
    for target in targets:
        if target not in predictor_raw.results:
            continue
        
        task_type = list(predictor_raw.results[target].values())[0].task_type
        
        raw_results = predictor_raw.results[target]
        pca_results = predictor_pca.results[target]
        
        # Get common models
        models = sorted(set(raw_results.keys()) & set(pca_results.keys()))
        
        if task_type == 'classification':
            raw_scores = [raw_results[m].f1 or 0 for m in models]
            pca_scores = [pca_results[m].f1 or 0 for m in models]
            metric = 'F1 Score'
        else:
            raw_scores = [raw_results[m].r2 or 0 for m in models]
            pca_scores = [pca_results[m].r2 or 0 for m in models]
            metric = 'R² Score'
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, raw_scores, width, label='Raw (1280)', color='steelblue')
        bars2 = ax.bar(x + width/2, pca_scores, width, label=f'PCA ({pca_dim})', color='darkorange')
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'{target}: Raw vs PCA Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.legend()
        ax.set_ylim(0, max(max(raw_scores), max(pca_scores)) * 1.15)
        
        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        safe_name = target.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '')
        plt.savefig(output_dir / f'{safe_name}_raw_vs_pca.png', dpi=150, bbox_inches='tight')
        plt.show()


def compare_pca_vs_raw(
    embedding_path: str = 'tf_embeddings_output/embeddings_1280.npz',
    target_csv: str = 'tf.csv',
    target_col: str = COL_TRANSFORMS,
    cv_folds: int = 5,
    n_iter: int = 10,
    pca_dims: list[int] = [64, 128, 256, 512],
):
    """
    Compare raw embeddings vs various PCA dimensions.
    Helps find optimal dimensionality for your data.
    
    Time: ~15-20 min per configuration (quick mode)
    Total: ~1-1.5 hours for full comparison
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Load data once
    data = np.load(embedding_path, allow_pickle=True)
    emb_genes = list(data['genes'])
    embeddings = data['embeddings']
    
    target_df = pd.read_csv(target_csv)
    control_mask = target_df[COL_GENE].str.lower() == 'control'
    
    if control_mask.sum() > 0:
        baseline = target_df.loc[control_mask, COL_AVG].mean()
        is_hit = (target_df[COL_PVAL] < 0.05) & (target_df[COL_AVG] > baseline)
        target_df[COL_TRANSFORMS] = is_hit.astype(float)
        target_df.loc[control_mask, COL_TRANSFORMS] = np.nan
        target_df = target_df[~control_mask].copy()
    
    gene_to_idx = {g: i for i, g in enumerate(emb_genes)}
    valid_mask = target_df[COL_GENE].isin(gene_to_idx.keys())
    target_df = target_df[valid_mask].copy()
    
    genes = target_df[COL_GENE].values
    X_raw = np.array([embeddings[gene_to_idx[g]] for g in genes])
    y = target_df[target_col].dropna().values.astype(int)
    X_raw = X_raw[~pd.isna(target_df[target_col].values)]
    
    print(f"\n{'='*70}")
    print("COMPARING: Raw Embeddings vs PCA Dimensions")
    print(f"{'='*70}")
    print(f"Samples: {X_raw.shape[0]}, Raw features: {X_raw.shape[1]}")
    print(f"Target: {target_col}")
    print(f"Testing: Raw + PCA dims {pca_dims}")
    print(f"{'='*70}\n")
    
    # Quick models for comparison
    models = {
        'LogisticRegression': LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    }
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []
    
    # Test raw
    print("Testing: Raw embeddings (1280-dim)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1', n_jobs=-1)
        results.append({
            'Dimensions': 'Raw (1280)',
            'Model': name,
            'F1_mean': scores.mean(),
            'F1_std': scores.std(),
        })
        print(f"  {name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Test PCA dimensions
    for n_comp in pca_dims:
        print(f"\nTesting: PCA ({n_comp}-dim)")
        
        pipeline_pre = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp, random_state=42))
        ])
        X_pca = pipeline_pre.fit_transform(X_raw)
        
        for name, model in models.items():
            scores = cross_val_score(model, X_pca, y, cv=cv, scoring='f1', n_jobs=-1)
            results.append({
                'Dimensions': f'PCA ({n_comp})',
                'Model': name,
                'F1_mean': scores.mean(),
                'F1_std': scores.std(),
            })
            print(f"  {name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("SUMMARY: Best F1 per Configuration")
    print(f"{'='*70}")
    
    summary = results_df.groupby('Dimensions').apply(
        lambda x: x.loc[x['F1_mean'].idxmax()]
    )[['Dimensions', 'Model', 'F1_mean', 'F1_std']]
    summary = summary.sort_values('F1_mean', ascending=False)
    
    for _, row in summary.iterrows():
        print(f"  {row['Dimensions']:15s} | {row['Model']:20s} | F1 = {row['F1_mean']:.3f} ± {row['F1_std']:.3f}")
    
    best = summary.iloc[0]
    print(f"\n★ RECOMMENDATION: Use {best['Dimensions']} with {best['Model']}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dims = ['Raw (1280)'] + [f'PCA ({d})' for d in pca_dims]
    x = np.arange(len(dims))
    width = 0.25
    
    for i, (name, _) in enumerate(models.items()):
        model_results = results_df[results_df['Model'] == name]
        means = [model_results[model_results['Dimensions'] == d]['F1_mean'].values[0] for d in dims]
        stds = [model_results[model_results['Dimensions'] == d]['F1_std'].values[0] for d in dims]
        ax.bar(x + i*width, means, width, yerr=stds, label=name, capsize=3)
    
    ax.set_xlabel('Feature Representation')
    ax.set_ylabel('F1 Score')
    ax.set_title('Raw Embeddings vs PCA: Model Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dims, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('pca_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: pca_comparison.png")
    plt.show()
    
    return results_df


if __name__ == '__main__':
    """
    RECOMMENDED WORKFLOW:
    
    Option 1: Quick comparison (~10-15 min)
        python predict_tuned.py --compare
    
    Option 2: Full pipeline with one config (~40-60 min)
        python predict_tuned.py
    
    Option 3: Full pipeline on BOTH raw and PCA (~80-120 min) [BEST]
        python predict_tuned.py --full
    """
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Quick comparison mode
        compare_pca_vs_raw(n_iter=10)
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Full comparison: raw vs PCA with all models
        run_full_comparison(
            embedding_path='tf_embeddings_output/embeddings_1280.npz',
            target_csv='tf.csv',
            output_dir='prediction_results',
            cv_folds=5,
            n_iter=20,
            pca_components=256,
            targets=[COL_TRANSFORMS, COL_AVG],
        )
    
    else:
        # Default: full pipeline with PCA only
        predictor = run_pipeline(
            embedding_path='tf_embeddings_output/embeddings_1280.npz',
            target_csv='tf.csv',
            output_dir='prediction_results',
            cv_folds=5,
            n_iter=20,
            use_pca=True,
            pca_components=256,
            targets=[COL_TRANSFORMS, COL_AVG],
        )