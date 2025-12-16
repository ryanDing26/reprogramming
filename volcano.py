"""Volcano Plot for TF Screening"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and filter data
df = pd.read_csv('tf_with_lengths.csv')
df = df[df['Well ID'].str.lower() != 'control']
df = df[df['sequence_length'] <= 1024].dropna(subset=['Average (%Myh6+ cells)', 'P-Value (%Myh6+ cells)'])

# Calculate -log10(p-value)
df['neg_log10_p'] = -np.log10(df['P-Value (%Myh6+ cells)'])

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Average (%Myh6+ cells)'], df['neg_log10_p'], alpha=0.5, s=20)
plt.xlabel('Average (%Myh6+ cells)')
plt.ylabel('-log₁₀(p-value)')
plt.title(f'Volcano Plot (95th Percentile) (n={len(df)}, ≤1024 AA)')
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)

# Focus on bulk of data (ignore extreme outliers); focuses on 95th percentile
plt.xlim(0, df['Average (%Myh6+ cells)'].quantile(0.95))
plt.ylim(0, df['neg_log10_p'].quantile(0.95))

plt.tight_layout()
plt.savefig('volcano_plot_95.png', dpi=300)
plt.show()

print(f"Plotted {len(df)} genes")