# pyscDblFinder

Python implementation of the scDblFinder workflow for doublet detection in
single-cell RNA-seq data, designed to run on AnnData/Scanpy objects.

This module mirrors the core ideas of the R package:
- optional pre-clustering
- artificial doublet generation
- iterative classifier training
- threshold optimization based on expected doublet rate

## What this package does

Given a count matrix in an `AnnData` object, pyscDblFinder estimates a
doublet score for each real cell and returns a final class (`doublet` or
`singlet`).

At a high level, the pipeline is:
1. Optional clustering of real cells (clustered mode).
2. Feature selection and artificial doublet generation.
3. Combined real + artificial embedding and KNN feature extraction.
4. Iterative XGBoost training and score refinement.
5. Final thresholding to obtain doublet calls.

Main entry point:
- `scDblFinder.py` -> `compute_doublet_score(...)`

## Repository layout (Python side)

- `scDblFinder.py`: main pipeline and model training
- `clustering.py`: fast clustering helper used in clustered mode
- `doublet_generation.py`: artificial doublet generation
- `misc.py`: feature engineering utilities (including cxds-like score)
- `thresholding.py`: doublet threshold optimization

## Requirements

Python:
- Python 3.9+

Core packages:
- `numpy`
- `pandas`
- `scipy`
- `anndata`
- `scanpy`
- `scikit-learn`
- `xgboost`

Optional GPU-related packages:
- `rapids-singlecell`
- `cuml`

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy anndata scanpy scikit-learn xgboost
```

If you want to import the module from scripts in this repository root,
this usually works directly because `scdblfinder/` is part of the repo.

## Input expectations

`compute_doublet_score(...)` expects an `AnnData` object:
- `adata.X` should contain raw counts (preferred), or
- `adata.layers['counts']` should contain raw counts.

Ground-truth labels are not required for running the method itself.
Ground truth is only needed for benchmark evaluation scripts.

## Quick start

### 1) Random mode (no clustering)

```python
import scanpy as sc
from scdblfinder.scDblFinder import compute_doublet_score

adata = sc.read_h5ad("your_data.h5ad")
adata_out = compute_doublet_score(
	adata,
	clusters_col=None,   # random mode
	n_iters=3,
	random_state=42,
	verbose=True,
)

print(adata_out.obs[["scDblFinder_score", "scDblFinder_class"]].head())
print("Threshold:", adata_out.uns.get("scDblFinder_threshold"))
```

### 2) Clustered mode (auto clustering)

```python
import scanpy as sc
from scdblfinder.scDblFinder import compute_doublet_score

adata = sc.read_h5ad("your_data.h5ad")
adata_out = compute_doublet_score(
	adata,
	clusters_col="clusters",  # if missing, fast clustering is run and stored here
	n_iters=3,
	random_state=42,
	verbose=True,
)
```

### 3) Clustered mode (precomputed clusters)

If your data already has cluster labels:

```python
adata.obs["my_clusters"] = ...
adata_out = compute_doublet_score(adata, clusters_col="my_clusters")
```

## Outputs

After running `compute_doublet_score(...)`, the returned object includes:

In `adata.obs`:
- `scDblFinder_score`: continuous doublet score (higher means more likely doublet)
- `scDblFinder_class`: final call (`doublet` or `singlet`)

In `adata.uns`:
- `scDblFinder_threshold`: threshold used to separate singlets and doublets

If `return_type='full'`, the returned object also includes artificial doublets.

## Important parameters

Commonly tuned parameters:
- `clusters_col`: set to `None` for random mode, or a column name for clustered mode
- `n_features`: number of selected genes used in feature selection
- `n_components`: number of PCA components
- `n_artificial`: override number of artificial doublets
- `prop_random`: fraction of random artificial doublets
- `n_iters`: iterative classifier refinement rounds
- `dbr`, `dbr_sd`, `dbr_per1k`: expected doublet-rate controls for thresholding
- `stringency`: threshold optimization aggressiveness
- `random_state`: reproducibility seed

## Reproducibility tips

- Fix `random_state` when comparing runs.
- Keep package versions stable (especially `scanpy`, `scikit-learn`, `xgboost`).
- Use the same preprocessing assumptions (counts in `adata.X` or `adata.layers['counts']`).

## Current scope and notes

- This is a Python implementation inspired by the R package behavior.
- Some low-level differences can remain due to backend/library differences.
- `samples_col` is currently accepted but ignored in the Python pipeline.

## Minimal troubleshooting

If results look unstable or unexpectedly weak:
- confirm counts are raw counts (not already transformed) where expected
- try both modes (`clusters_col=None` and clustered mode)
- verify `xgboost` installation and version
- run with `verbose=True` to inspect each stage

If clustering looks poor, test with your own precomputed clusters and pass them via `clusters_col`.