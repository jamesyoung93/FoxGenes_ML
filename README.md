# FOX gene candidate prediction with multi-omic machine learning

This repository contains the machine-learning pipeline from the published FOX gene study in *Scientific Reports*. It integrates nitrogen step-down transcriptomics, quantitative proteomics, promoter architecture, genomic context, Gene Ontology, and reciprocal-best-hit conservation to rank genes associated with oxic nitrogen fixation in *Anabaena* sp. PCC 7120. The goal is candidate prioritization for experimental follow-up, not a claim that an untested gene is FOX-essential.

**Paper:** [Predicting FOX gene candidates for oxic nitrogen fixation using multi-omic machine learning and comparative bioinformatics](https://doi.org/10.1038/s41598-026-41873-w), *Scientific Reports* 16, 11412 (2026)

**Interactive app:** [FOX Gene Complement Explorer](https://foxgenesapp-h6mlszof5jdd6p2bpq22wr.streamlit.app/)

## Key result

Across 20 repeated stratified 80/20 train-test splits, the paper reports ROC-AUC values up to 0.80, average precision up to 0.55, and precision of 0.39 among the top 20 ranked genes, compared with a 0.075 positive-class prevalence. The released pFOX scores are ranking statistics and are not calibrated probabilities of biological essentiality.

## Methods and stack

- Logistic regression, random forest, and XGBoost classifiers
- Repeated stratified holdout evaluation and location-blocked sensitivity analysis
- SHAP-based interpretation of nonlinear predictors
- Python, pandas, NumPy, scikit-learn, XGBoost, and matplotlib

## Install

```bash
git clone https://github.com/jamesyoung93/FoxGenes_ML.git
cd FoxGenes_ML
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Repository structure

```text
FoxGenes_ML/
|-- src/                           # Data preparation, feature sets, and evaluation
|-- data/                          # Published modeling table
|-- outputs/                       # Manuscript metrics, models, and ranked predictions
|-- run_bootstrap_ablation_suite.py
|-- run_blocked_area_cv.py
|-- train_final_and_predict_unknowns.py
|-- make_fig5_6panel.py
`-- requirements.txt
```

## Reproduce the published analyses

### Repeated holdout evaluation

```bash
python run_bootstrap_ablation_suite.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --out outputs/table1_check \
  --train_fracs 0.8 \
  --n_splits 20
```

Precomputed Table 1 and Supplementary Table S5 results are in `outputs/bootstrap_ablation/`.

### Location-blocked cross-validation

```bash
python run_blocked_area_cv.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression \
  --n_groups 5 \
  --n_partitions 4 \
  --out outputs/blocked_cv_check
```

### Train final models and rank unlabeled genes

```bash
python train_final_and_predict_unknowns.py \
  --data data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression \
  --out outputs/final_check \
  --n_iter 60 \
  --cv_folds 5
```

### Rebuild the SHAP figure

```bash
python make_fig5_6panel.py \
  --model_joblib outputs/final_predictions/final_model_xgb.joblib \
  --feature_matrix data/prepared_for_modeling.plus_conservation_v2.csv \
  --predictions_all outputs/final_predictions/predictions_all_ensemble.csv \
  --outdir outputs/fig5
```

## Feature sets

| Name | Description |
|---|---|
| `principled_expression` | RNA-seq, proteomics, promoter, conservation, Gene Ontology, and genomic position. This is the primary configuration. |
| `principled_expression_no_position` | The same evidence without genomic coordinates. |
| `baseline_full` | All available features. |

## Data availability

The prepared modeling table and precomputed manuscript outputs are included. The underlying RNA-seq, proteomics, transcription-start-site, and annotation sources are public and are identified in the paper's Data Availability section. Comparative conservation features can be rebuilt with [cyanobacteria-diazotrophic-proteome](https://github.com/jamesyoung93/cyanobacteria-diazotrophic-proteome).

The repository includes trained model artifacts larger than 5 MB for reproducibility. No file in the current release exceeds GitHub's 100 MB file limit.

## Related repositories

- [FoxGenesApp](https://github.com/jamesyoung93/FoxGenesApp): interactive filtering and size-constrained gene-complement design
- [cyanobacteria-diazotrophic-proteome](https://github.com/jamesyoung93/cyanobacteria-diazotrophic-proteome): comparative proteomics and reciprocal-best-hit pipeline

## Citation

```bibtex
@article{Young2026FoxGenes,
  author  = {Young, James and Gu, Liping and Zhou, Ruanbao},
  title   = {Predicting FOX gene candidates for oxic nitrogen fixation using multi-omic machine learning and comparative bioinformatics},
  journal = {Scientific Reports},
  year    = {2026},
  volume  = {16},
  pages   = {11412},
  doi     = {10.1038/s41598-026-41873-w}
}
```

## License

Code is available under the [MIT License](LICENSE). Source datasets and third-party materials retain their original terms.
