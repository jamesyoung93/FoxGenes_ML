# FoxGenes_ML

Machine learning pipeline for predicting FOX (Functioning under OXic conditions) gene candidates in *Anabaena* sp. PCC 7120, enabling nitrogen fixation in the presence of oxygen.

## Overview

This repository contains the complete codebase, data, and outputs for training and evaluating machine learning models that predict FOX gene candidates. The pipeline uses multi-omic features including transcriptomics, proteomics, promoter architecture, and comparative genomics.

**Associated Publication:** Young, J.T. et al. "Predicting FOX gene candidates for oxic nitrogen fixation using multi-omic machine learning and comparative bioinformatics" (2026). *Nature Scientific Reports*.

## Quick Start (Linux/HPC Environment)

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/jamesyoung93/FoxGenes_ML.git
cd FoxGenes_ML

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "from src.feature_sets import FEATURE_SETS; print([fs.name for fs in FEATURE_SETS])"
```

### 3. Reproduce Key Results

```bash
# Reproduce Table 1 metrics (nested CV evaluation)
python run_nested_cv.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression_no_position \
  --out my_outputs/nested_cv

# Or simply examine the pre-computed outputs
cat outputs/nested_cv/summary.csv
```

## Repository Structure

```
FoxGenes_ML/
├── src/                              # Core Python module
│   ├── __init__.py
│   ├── data_prep.py                  # Data loading and preprocessing
│   ├── feature_sets.py               # Feature set definitions
│   ├── nested_cv.py                  # Nested cross-validation
│   ├── bootstrap_holdout.py          # Bootstrap holdout evaluation
│   └── blocked_area_cv.py            # Location-blocked CV
│
├── data/
│   └── prepared_for_modeling.plus_conservation_v2.csv  # Feature matrix
│
├── outputs/                          # Pre-computed manuscript outputs
│   ├── nested_cv/                    # Table 1 performance metrics
│   │   ├── summary.csv               # Mean ± SD across 20 splits
│   │   ├── fold_metrics.csv          # Per-split metrics
│   │   └── ensemble/                 # Ensemble model results
│   ├── bootstrap_ablation/           # S5: Feature ablation study
│   ├── blocked_cv/                   # S6: Location-blocked CV results
│   └── final_predictions/            # Trained models and pFOX scores
│       ├── final_model_xgb.joblib    # Trained XGBoost model
│       ├── final_model_rf.joblib     # Trained Random Forest model
│       ├── final_params_*.json       # Tuned hyperparameters
│       └── predictions_*.csv         # pFOX scores for all genes
│
├── run_nested_cv.py                  # Nested CV evaluation
├── run_bootstrap_ablation_suite.py   # Feature ablation study
├── run_blocked_area_cv.py            # Location-blocked CV
├── run_bootstrap_holdout.py          # Single bootstrap run
├── train_final_and_predict_unknowns.py  # Train final models
├── fig5_shap_examples.py             # SHAP analysis
├── make_fig5_6panel.py               # Figure 5 with interactions
│
├── legacy/                           # Archived previous implementation
├── requirements.txt
├── LICENSE
└── README.md
```

## Pre-Computed Outputs

This repository includes the actual outputs used in the manuscript, allowing you to examine results directly or verify reproducibility.

### Table 1 Metrics (`outputs/nested_cv/`)

The `summary.csv` file contains mean ± SD performance metrics across 20 repeated stratified holdout splits. These are the values reported in Table 1 of the manuscript.

### Supplementary Table S5 (`outputs/bootstrap_ablation/`)

Feature ablation results comparing model performance across different feature configurations. The `ablation_comparison_summary.csv` shows how removing different feature categories affects predictive performance.

### Supplementary Table S6 (`outputs/blocked_cv/`)

Location-blocked cross-validation results testing whether models generalize across genomic regions. The genome was partitioned into 5 contiguous blocks with 4 partition offsets, yielding 20 total splits.

### Final Predictions (`outputs/final_predictions/`)

Trained models and pFOX (probability of FOX) scores for all 5,054 genes. The `predictions_all_ensemble.csv` contains the ensemble predictions used for downstream analysis and the interactive web application.

## Detailed Usage

### Running Nested Cross-Validation

This is the primary evaluation approach. It uses 20 repeated stratified 80/20 splits with nested hyperparameter tuning via 3-fold inner CV.

```bash
python run_nested_cv.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression_no_position \
  --out outputs/my_nested_cv \
  --n_splits 20 \
  --inner_folds 3 \
  --n_iter 30 \
  --n_jobs 4
```

The nested structure ensures that reported test-set metrics are unbiased by hyperparameter selection. Within each of the 20 outer splits, hyperparameters are tuned using only the training portion, then evaluated on the held-out test set.

### Running Bootstrap Ablation Study

Compares performance across different feature configurations to understand which features contribute most to predictive power.

```bash
python run_bootstrap_ablation_suite.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --out outputs/my_ablation \
  --train_fracs 0.8 \
  --n_splits 20 \
  --n_jobs 4
```

### Running Location-Blocked CV

Tests whether models exploit spatial clustering of labels by holding out contiguous genomic regions.

```bash
python run_blocked_area_cv.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression \
  --n_groups 5 \
  --n_partitions 4 \
  --out outputs/my_blocked_cv
```

This divides the chromosome into 5 contiguous blocks based on genomic coordinates. Block boundaries are shifted 4 times using evenly-spaced offsets to generate 20 total evaluation splits.

### Training Final Models

After validation, train on all 903 labeled genes and generate predictions for unknowns.

```bash
python train_final_and_predict_unknowns.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression_no_position \
  --out outputs/my_final \
  --n_iter 60 \
  --cv_folds 5
```

### Generating SHAP Figures

Create SHAP analysis plots including interaction effects (Figure 5).

```bash
python make_fig5_6panel.py \
  --model_joblib outputs/final_predictions/final_model_xgb.joblib \
  --feature_matrix data/prepared_for_modeling.plus_conservation_v2.csv \
  --predictions_all outputs/final_predictions/predictions_all_ensemble.csv \
  --outdir outputs/fig5 \
  --dpi 600
```

## Feature Sets

The pipeline supports multiple feature configurations. The `principled_expression_no_position` set (excluding genomic coordinates) is used for main-text results to avoid potential positional leakage.

| Feature Set | Description |
|-------------|-------------|
| `baseline_full` | All 92 available features |
| `principled_expression` | RPKM + proteomics + promoter + conservation + position |
| `principled_expression_no_position` | Same as above, excluding chromosome coordinates |
| `principled_expression_no_position_plus_conservation` | Emphasizes conservation features |

## Methodology

### Two-Stage Workflow

**Stage 1 (Evaluation):** Models are evaluated using 20 repeated stratified 80/20 splits. Within each split, hyperparameters for Random Forest and XGBoost are tuned on the training portion using 3-fold inner cross-validation with RandomizedSearchCV (30 iterations, scoring by average precision).

**Stage 2 (Final Prediction):** After validation, classifiers are retrained on all 903 labeled genes using 5-fold CV with 60 RandomizedSearchCV iterations. Final models generate pFOX scores for the 4,151 unlabeled genes.

### Models

Three classifiers are trained and ensembled. Logistic Regression uses L2 regularization with stepwise feature selection (Wald entry p<0.01, exit p>0.05). Random Forest uses balanced class weights with tuned hyperparameters for n_estimators, max_depth, max_features, and min_samples. XGBoost uses binary logistic objective with scale_pos_weight for class imbalance and tuned hyperparameters.

The ensemble probability is the arithmetic mean of all three model probabilities.

### SHAP Analysis

SHAP values are computed using XGBoost's native TreeSHAP implementation (`pred_contribs=True`). For interaction analysis in Figure 5 panels e-f, `pred_interactions=True` decomposes pairwise feature contributions. All SHAP values are in log-odds space.

## Data Description

The feature matrix contains 5,054 genes from *Anabaena* sp. PCC 7120 with 68 validated FOX genes (positive class), 835 conserved non-essential genes (negative proxy class), and 4,151 unlabeled genes.

Features include RPKM expression values at 0h, 12h, and 21h post nitrogen step-down, proteomics fold-change values from mass spectrometry, divergent promoter distance and operon membership, conservation features from comparative proteomics (presence/absence of homologs in diazotrophs vs non-diazotrophs), and genomic coordinates.

## HPC Job Submission

Example SLURM script for cluster environments:

```bash
#!/bin/bash
#SBATCH --job-name=foxgenes
#SBATCH --output=foxgenes_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load python/3.10
source /path/to/FoxGenes_ML/.venv/bin/activate

python run_nested_cv.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression_no_position \
  --out outputs/nested_cv \
  --n_jobs 8
```

## Related Repositories

The [FoxGenesApp](https://github.com/jamesyoung93/FoxGenesApp) repository contains an interactive Streamlit application for exploring predictions. The [cyanobacteria-diazotrophic-proteome](https://github.com/jamesyoung93/cyanobacteria-diazotrophic-proteome) repository contains the comparative proteomics pipeline used to generate conservation features.

## Citation

```
Young, J.T. et al. (2026). Predicting FOX gene candidates for oxic nitrogen fixation 
using multi-omic machine learning and comparative bioinformatics. 
Nature Scientific Reports. [DOI pending]
```

## License

MIT License - see [LICENSE](LICENSE) for details.
