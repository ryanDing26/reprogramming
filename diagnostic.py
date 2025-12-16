import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

# Load data
data = np.load('tf_embeddings_output/embeddings_1280.npz', allow_pickle=True)
genes, embeddings = data['genes'], data['embeddings']
phenotype_df = pd.read_csv('tf_with_lengths.csv')

# Match
gene_to_target = dict(zip(phenotype_df['Well ID'], phenotype_df['Average (%Myh6+ cells)']))
valid_idx, y = zip(*[(i, gene_to_target[g]) for i, g in enumerate(genes) 
                      if g in gene_to_target and pd.notna(gene_to_target[g])])
X, y = embeddings[list(valid_idx)], np.array(y)
print(f"Matched {len(y)} samples")

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X)

# Option 1: Log-transform for coloring
y_log = np.log1p(y)  # log(1 + y) to handle zeros

# Option 2: Rank/percentile transform (often better for skewed data)
y_rank = np.argsort(np.argsort(y)) / len(y)  # 0 to 1 percentile

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log-scaled colors
sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=y_log, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(sc1, ax=axes[0], label='log(%Myh6+ + 1)')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('Log-scaled colors')

# Rank-scaled colors
sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=y_rank, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(sc2, ax=axes[1], label='Percentile')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[1].set_title('Rank-scaled colors')

plt.tight_layout()
plt.savefig('pca_colored_rescaled.png', dpi=150)
plt.show()

# Stats
for i, name in enumerate(['PC1', 'PC2']):
    r, p = spearmanr(coords[:, i], y)
    print(f"{name}: r={r:.3f}, p={p:.2e}")