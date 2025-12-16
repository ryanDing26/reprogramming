"""
Simple Prediction Heads for TF Screening
=========================================
- ElasticNet, SVR, MLP regression
- 1280-dim ESM2 embeddings
- K-fold CV (k=3,4,5) with hyperparameter tuning
- Sequences > 1024 AA excluded
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data(
    embedding_path: str,
    target_csv: str,
    target_col: str,
    gene_col: str = 'Well ID',
    max_seq_len: int = 1024,
):
    """
    Load embeddings and targets, filter by sequence length.
    
    Args:
        embedding_path: Path to .npz with 'genes', 'embeddings', 'lengths'
        target_csv: Path to CSV with targets
        target_col: Column name for target variable
        gene_col: Column name for gene identifiers
        max_seq_len: Maximum sequence length (drop longer)
    
    Returns:
        X: (n_samples, 1280) embeddings
        y: (n_samples,) target values
        genes: gene names
    """
    # Load embeddings
    data = np.load(embedding_path, allow_pickle=True)
    emb_genes = list(data['genes'])
    embeddings = data['embeddings']
    
    # Get sequence lengths if available
    if 'lengths' in data:
        lengths = data['lengths']
    else:
        # If no lengths, assume all pass
        lengths = np.ones(len(emb_genes)) * max_seq_len
    
    # Filter by sequence length
    len_mask = lengths <= max_seq_len
    emb_genes = [g for g, m in zip(emb_genes, len_mask) if m]
    embeddings = embeddings[len_mask]
    
    print(f"Loaded {len(emb_genes)} embeddings (dropped {(~len_mask).sum()} with len > {max_seq_len})")
    
    # Load targets
    df = pd.read_csv(target_csv)
    
    # Remove controls
    control_mask = df[gene_col].str.lower() == 'control'
    df = df[~control_mask].copy()
    
    # Align with embeddings
    gene_to_idx = {g: i for i, g in enumerate(emb_genes)}
    valid_genes = df[gene_col].isin(gene_to_idx.keys())
    df = df[valid_genes].copy()
    
    # Extract aligned data
    genes = df[gene_col].values
    X = np.array([embeddings[gene_to_idx[g]] for g in genes])
    y = df[target_col].values
    
    # Remove NaN targets
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    genes = genes[valid_mask]
    
    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target '{target_col}': mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, genes


def get_models():
    """
    Return models with hyperparameter grids.
    """
    models = {
        'ElasticNet': {
            'model': ElasticNet(max_iter=5000, random_state=42),
            'params': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'model__C': [0.1, 1.0, 10.0, 100.0],
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto'],
                'model__epsilon': [0.01, 0.1, 0.5],
            }
        },
        'MLP': {
            'model': MLPRegressor(
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
            'params': {
                'model__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
                'model__alpha': [0.001, 0.01, 0.1],
                'model__learning_rate_init': [0.001, 0.01],
            }
        },
    }
    return models


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: list[int] = [3, 4, 5],
    n_jobs: int = -1,
):
    """
    Train all models with k-fold CV and hyperparameter tuning.
    
    Args:
        X: Features (n_samples, 1280)
        y: Targets (n_samples,)
        k_folds: List of k values for CV
        n_jobs: Parallel jobs (-1 = all cores)
    
    Returns:
        results: Dict with all results
        best_model: Tuple (name, pipeline, params) of best model
    """
    models = get_models()
    results = []
    
    for k in k_folds:
        print(f"\n{'='*60}")
        print(f"K-FOLD CV: k={k}")
        print(f"{'='*60}")
        
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        
        for model_name, config in models.items():
            print(f"\n--- {model_name} ---")
            
            # Build pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', config['model']),
            ])
            
            # Grid search
            grid = GridSearchCV(
                pipeline,
                param_grid=config['params'],
                cv=cv,
                scoring='r2',
                n_jobs=n_jobs,
                refit=True,
            )
            
            grid.fit(X, y)
            
            # Get CV predictions with best model
            y_pred = cross_val_predict(grid.best_estimator_, X, y, cv=cv)
            
            # Metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            print(f"  Best params: {grid.best_params_}")
            print(f"  R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")
            
            results.append({
                'model': model_name,
                'k': k,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'best_params': grid.best_params_,
                'predictions': y_pred,
                'pipeline': grid.best_estimator_,
            })
    
    # Find overall best
    best_result = max(results, key=lambda x: x['r2'])
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_result['model']} (k={best_result['k']})")
    print(f"  R² = {best_result['r2']:.4f}")
    print(f"  Params: {best_result['best_params']}")
    print(f"{'='*60}")
    
    return results, best_result


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    best_result: dict,
    save_path: str = None,
):
    """
    Train final model on all data with best hyperparameters.
    
    Args:
        X: All features
        y: All targets
        best_result: Result dict from train_and_evaluate
        save_path: Path to save model (optional)
    
    Returns:
        Fitted pipeline
    """
    print(f"\nTraining final {best_result['model']} on all {len(y)} samples...")
    
    # Get model with best params
    models = get_models()
    config = models[best_result['model']]
    
    # Extract actual params (remove 'model__' prefix)
    params = {k.replace('model__', ''): v for k, v in best_result['best_params'].items()}
    
    # Build and fit
    model = config['model'].__class__(**{**model.get_params(), **params})
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])
    
    pipeline.fit(X, y)
    
    if save_path:
        joblib.dump({
            'pipeline': pipeline,
            'model_name': best_result['model'],
            'params': best_result['best_params'],
            'r2': best_result['r2'],
        }, save_path)
        print(f"Saved to: {save_path}")
    
    return pipeline


def plot_results(results: list, y_true: np.ndarray, output_dir: str = None):
    """
    Plot comparison of models and predictions.
    """
    output_dir = Path(output_dir) if output_dir else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. R² comparison across models and k values
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['ElasticNet', 'SVR', 'MLP']
    k_values = sorted(set(r['k'] for r in results))
    x = np.arange(len(models))
    width = 0.25
    
    for i, k in enumerate(k_values):
        r2_scores = [next(r['r2'] for r in results if r['model'] == m and r['k'] == k) for m in models]
        bars = ax.bar(x + i * width, r2_scores, width, label=f'k={k}')
        
        for bar, score in zip(bars, r2_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Comparison Across K-Fold Values')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, max(r['r2'] for r in results) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150)
    plt.show()
    
    # 2. Predicted vs Actual for best model
    best = max(results, key=lambda x: x['r2'])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, best['predictions'], alpha=0.5, edgecolors='k', linewidth=0.3)
    
    lims = [min(y_true.min(), best['predictions'].min()),
            max(y_true.max(), best['predictions'].max())]
    ax.plot(lims, lims, 'r--', label='Perfect')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f"{best['model']} (k={best['k']})\nR²={best['r2']:.4f}, RMSE={best['rmse']:.4f}")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_model_predictions.png', dpi=150)
    plt.show()
    
    # 3. Residuals
    residuals = y_true - best['predictions']
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Count')
    ax.set_title(f"Residual Distribution\nMean={residuals.mean():.3f}, Std={residuals.std():.3f}")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=150)
    plt.show()


def run_prediction_head(
    embedding_path: str,
    target_csv: str,
    target_col: str,
    output_dir: str,
    gene_col: str = 'Well ID',
    max_seq_len: int = 1024,
    k_folds: list[int] = [3, 4, 5],
):
    """
    Run complete prediction pipeline for one target column.
    
    Args:
        embedding_path: Path to embeddings .npz
        target_csv: Path to target CSV
        target_col: Column name to predict
        output_dir: Output directory
        gene_col: Gene identifier column
        max_seq_len: Max sequence length filter
        k_folds: K values for CV
    
    Returns:
        results, best_result, final_pipeline
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"PREDICTION HEAD: {target_col}")
    print(f"{'#'*60}")
    
    # Load data
    X, y, genes = load_data(
        embedding_path=embedding_path,
        target_csv=target_csv,
        target_col=target_col,
        gene_col=gene_col,
        max_seq_len=max_seq_len,
    )
    
    # Train and evaluate
    results, best_result = train_and_evaluate(X, y, k_folds=k_folds)
    
    # Save results summary
    summary_df = pd.DataFrame([{
        'Model': r['model'],
        'K': r['k'],
        'R2': r['r2'],
        'RMSE': r['rmse'],
        'MAE': r['mae'],
    } for r in results])
    summary_df.to_csv(output_dir / 'cv_results.csv', index=False)
    
    # Train final model
    model_path = output_dir / 'final_model.joblib'
    final_pipeline = train_final_model(X, y, best_result, save_path=str(model_path))
    
    # Plots
    plot_results(results, y, output_dir)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'gene': genes,
        'actual': y,
        'predicted': best_result['predictions'],
        'residual': y - best_result['predictions'],
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    print(f"\nOutputs saved to: {output_dir}/")
    
    return results, best_result, final_pipeline


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Target columns to predict
    TARGET_COLS = [
        'Average (%Myh6+ cells)'
    ]
    
    # Run separate prediction head for each target
    for target_col in TARGET_COLS:
        safe_name = target_col.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '')
        output_dir = f'results_{safe_name}'
        
        try:
            run_prediction_head(
                embedding_path='tf_embeddings_output/embeddings_1280.npz',
                target_csv='tf.csv',
                target_col=target_col,
                output_dir=output_dir,
                max_seq_len=1024,
                k_folds=[3, 4, 5],
            )
        except Exception as e:
            print(f"\nFailed for {target_col}: {e}")