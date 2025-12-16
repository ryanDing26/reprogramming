"""
Generates distribution of Average (%Myh6+ cells) and sequence length columns
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('tf_with_lengths.csv')

# Figure 1: %Myh6+ cells
plt.figure(figsize=(6, 5))
myh6_data = df['Average (%Myh6+ cells)'].dropna()
plt.hist(myh6_data, bins=50, edgecolor='black')

# Tight x-axis limits with small padding
x_min, x_max = myh6_data.min(), myh6_data.max()
x_range = x_max - x_min
plt.xlim(x_min - 0.05*x_range, x_max + 0.05*x_range)

plt.xlabel('Average (%Myh6+ cells)')
plt.ylabel('Count')
plt.title('Distribution of %Myh6+ cells')
plt.tight_layout()
plt.savefig('myh6_distribution.png', dpi=150)

# Figure 2: Sequence lengths
plt.figure(figsize=(6, 5))
length_data = df['sequence_length'].dropna()
plt.hist(length_data, bins=50, edgecolor='black')

# Tight x-axis limits with small padding
x_min, x_max = length_data.min(), length_data.max()
x_range = x_max - x_min
plt.xlim(x_min - 0.05*x_range, x_max + 0.05*x_range)

plt.xlabel('Sequence Length (aa)')
plt.ylabel('Count')
plt.title('Distribution of Protein Sequence Lengths')
plt.tight_layout()
plt.savefig('sequence_length_distribution.png', dpi=150)

plt.show()

# Print statistics
print("\n" + "="*60)
print("STATISTICS: Average (%Myh6+ cells)")
print("="*60)
print(f"Count:        {len(myh6_data)}")
print(f"Missing:      {df['Average (%Myh6+ cells)'].isna().sum()}")
print(f"Min:          {myh6_data.min():.2f}%")
print(f"Max:          {myh6_data.max():.2f}%")
print(f"Mean:         {myh6_data.mean():.2f}%")
print(f"Median:       {myh6_data.median():.2f}%")
print(f"Std Dev:      {myh6_data.std():.2f}%")
print(f"25th percentile: {myh6_data.quantile(0.25):.2f}%")
print(f"75th percentile: {myh6_data.quantile(0.75):.2f}%")
print(f"IQR:          {myh6_data.quantile(0.75) - myh6_data.quantile(0.25):.2f}%")
print(f"Skewness:     {myh6_data.skew():.2f}")

print("\n" + "="*60)
print("STATISTICS: Sequence Length (aa)")
print("="*60)
print(f"Count:        {len(length_data)}")
print(f"Missing:      {df['sequence_length'].isna().sum()}")
print(f"Min:          {length_data.min():.0f} aa")
print(f"Max:          {length_data.max():.0f} aa")
print(f"Mean:         {length_data.mean():.1f} aa")
print(f"Median:       {length_data.median():.1f} aa")
print(f"Std Dev:      {length_data.std():.1f} aa")
print(f"25th percentile: {length_data.quantile(0.25):.1f} aa")
print(f"75th percentile: {length_data.quantile(0.75):.1f} aa")
print(f"IQR:          {length_data.quantile(0.75) - length_data.quantile(0.25):.1f} aa")
print(f"Skewness:     {length_data.skew():.2f}")
print("="*60 + "\n")