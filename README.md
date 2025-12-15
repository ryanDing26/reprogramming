# ESM-Based Protein Prediction Pipeline for Cardiac Reprogramming

Predict cardiac reprogramming efficiency (%Myh6+ cells) from protein sequences using ESM-2 (Evolutionary Scale Modeling) embeddings.

## Overview

This pipeline:
1. Fetches protein sequences from NCBI for your gene list
2. Generates high-dimensional embeddings using Meta's ESM-2 protein language model
3. Trains multiple ML models to predict %Myh6+ cells
4. Evaluates performance and generates visualizations

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

**Note:** First run will download the ESM-2 model (~2.5GB for 650M parameter model)

### 1. Protein Sequences → ESM Embeddings

ESM-2 is a transformer-based protein language model trained on 65M protein sequences. It learns rich representations of protein structure and function.

**What it captures:**
- Amino acid composition patterns
- Local sequence motifs
- Long-range interactions
- Evolutionary conservation
- Structural properties

## Model Variants

You can change the ESM model in the code:

Available models:
- `esm2_t30_150M_UR50D` - 150M params (faster, 640-dim embeddings)
- `esm2_t33_650M_UR50D` - 650M params (default, 1280-dim embeddings)
- `esm2_t36_3B_UR50D` - 3B params (best quality, 2560-dim embeddings, requires 16GB+ RAM)

## Citation

If using ESM-2:
```
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and others},
  journal={bioRxiv},
  year={2022}
}
```

## Further Reading

- [ESM GitHub](https://github.com/facebookresearch/esm)
- [ESM Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
- [Protein Language Models Review](https://www.nature.com/articles/s41592-022-01490-z)