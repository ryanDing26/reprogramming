import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

# Load embeddings
data = np.load('tf_embeddings_output/embeddings.npz', allow_pickle=True)
genes = data['genes']
embeddings = data['embeddings']  # shape: (n_genes, 640)

# PCA
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(embeddings)

# UMAP
umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_coords = umap_model.fit_transform(embeddings)

# PCA figure
plt.figure(figsize=(7, 6))
plt.scatter(pca_coords[:, 0], pca_coords[:, 1], s=80, alpha=0.7)
# for i, gene in enumerate(genes):
#     plt.annotate(gene, (pca_coords[i, 0], pca_coords[i, 1]), fontsize=8)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA')
plt.tight_layout()
plt.savefig('pca.png', dpi=150)
plt.show()


# UMAP figure
plt.figure(figsize=(7, 6))
plt.scatter(umap_coords[:, 0], umap_coords[:, 1], s=80, alpha=0.7)
# for i, gene in enumerate(genes):
#     plt.annotate(gene, (umap_coords[i, 0], umap_coords[i, 1]), fontsize=8)

plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP')
plt.tight_layout()
plt.savefig('umap.png', dpi=150)
plt.show()
