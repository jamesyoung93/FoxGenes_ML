# FoxGenes_ML

Machine learning pipeline for predicting FOX (Functioning under OXic conditions) gene candidates in *Anabaena* sp. PCC 7120.

**Associated Publication:** Young, J.T., Gu, L. & Zhou, R. "Predicting FOX gene candidates for oxic nitrogen fixation using multi-omic machine learning and comparative bioinformatics." *Nature Scientific Reports* (submitted).

## Setup

```bash
git clone https://github.com/jamesyoung93/FoxGenes_ML.git
cd FoxGenes_ML

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure

```
FoxGenes_ML/
├── src/                           # Core module
│   ├── data_prep.py               # Data loading
│   ├── feature_sets.py            # Feature set definitions
│   ├── nested_cv.py               # Nested CV (not used for Table 1)
│   ├── bootstrap_holdout.py       # Repeated holdout evaluation (Table 1)
│   └── blocked_area_cv.py         # Location-blocked CV (Table S6)
├── data/
│   └── prepared_for_modeling.plus_conservation_v2.csv
├── outputs/                       # Manuscript results
│   ├── bootstrap_ablation/        # Table 1 & S5 metrics
│   ├── blocked_cv/                # Table S6 metrics
│   └── final_predictions/         # Trained models & pFOX scores
├── run_bootstrap_ablation_suite.py
├── run_blocked_area_cv.py
├── train_final_and_predict_unknowns.py
├── make_fig5_6panel.py
└── requirements.txt
```

## Reproducing Results

### Table 1: Model Performance (repeated stratified holdout)

Table 1 reports metrics from 20 repeated stratified 80/20 train-test splits using the **principled_expression** feature set (includes genomic position).

```bash
python run_bootstrap_ablation_suite.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --out outputs/table1_check \
  --train_fracs 0.8 \
  --n_splits 20
```

Pre-computed results are in `outputs/bootstrap_ablation/principled_expression/summary.csv`.

### Table S6: Location-Blocked CV

Tests generalization across genomic regions (5 blocks × 4 partition offsets = 20 splits).

```bash
python run_blocked_area_cv.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression \
  --n_groups 5 \
  --n_partitions 4 \
  --out outputs/blocked_cv_check
```

### Final Models & Predictions

After evaluation, models were retrained on all 903 labeled genes with 5-fold CV hyperparameter tuning (60 iterations).

```bash
python train_final_and_predict_unknowns.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression \
  --out outputs/final_check \
  --n_iter 60 \
  --cv_folds 5
```

Final trained models and genome-wide pFOX scores are in `outputs/final_predictions/`.

### Figure 5: SHAP Analysis

```bash
python make_fig5_6panel.py \
  --model_joblib outputs/final_predictions/final_model_xgb.joblib \
  --feature_matrix data/prepared_for_modeling.plus_conservation_v2.csv \
  --predictions_all outputs/final_predictions/predictions_all_ensemble.csv \
  --outdir outputs/fig5
```

## Feature Sets

| Name | Description |
|------|-------------|
| `principled_expression` | RPKM + proteomics + promoter + conservation + position (**main model**) |
| `principled_expression_no_position` | Same without genomic coordinates (ablation) |
| `baseline_full` | All available features |

## Related Repositories

- [FoxGenesApp](https://github.com/jamesyoung93/FoxGenesApp) — Interactive web tool for exploring predictions
- [cyanobacteria-diazotrophic-proteome](https://github.com/jamesyoung93/cyanobacteria-diazotrophic-proteome) — RBH comparative proteomics pipeline

## License

MIT
