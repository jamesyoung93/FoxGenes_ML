# FoxGenes_ML

## Data Sources / References

1. **Lechno-Yossef et al., 2011**  
   *Title:* *Identification of ten Anabaena sp. genes that under aerobic conditions are required for growth on dinitrogen but not for growth on fixed nitrogen*  
   *Journal:* Journal of Bacteriology  
   - Supplementary Table of this work provides about 75 experimentally validated FOX genes.  
   - Established ~10 additional FOX genes via insertional mutagenesis.  

2. **Tiruveedula & Wangikar, 2017**  
   *Title:* *Gene essentiality, conservation index and co-evolution of genes in cyanobacteria*  
   *Journal:* PLOS One  
   - Findings of non-essential genes, many conserved.  
   - Introduced the concept of NotFOX genes so the model could have positive and negative examples.  

3. **Flaherty et al., 2011**  
   *Title:* *Directional RNA deep sequencing sheds new light on the transcriptional response of Anabaena sp. strain PCC 7120 to combined-nitrogen deprivation*  
   *Journal:* BMC Genomics  
   - Time-series RNA-Seq under N-deprivation: 0 h, 6 h, 12 h, 21 h.  
   - Key input for transcriptomic-based FOX prediction.  

4. **Zhang et al., 2021**  
   *Title:* *Quantitative Proteomics Reveals the Protein Regulatory Network of Anabaena sp. PCC 7120 under Nitrogen Deficiency*  
   *Journal:* Journal of Proteome Research  
   - Provided protein fold-changes and p-values under N-starvation.  

---

## Methods

### Data Gathering  
Data from prior transcriptomic and proteomic experiments on *Anabaena sp.* PCC 7120 were gathered from the sources above. These data frames were joined based on gene names present in both datasets. This bioinformatic data was further joined with experimentally validated FOX genes from Lechno-Yossef et al., 2011. The remaining genes were either marked as **“Unknown”** (may or may not be a FOX gene) or **“NotFOX”**. Determination of **“NotFOX”** genes was made from a study of non-essential genes that are conserved across cyanobacteria in general (Tiruveedula & Wangikar, 2017), providing negative examples for model training.

### Data Cleaning and Feature Engineering  
Some genes were missing information from transcriptomic and proteomic datasets, which also led to missing derived variables such as fold change. These missing-data situations were handled by:
- Creating a binned representation of the original read counts or fold changes, and  
- Filling zeros for any missing values in the original columns (so that no genes were deleted).  

Once missing data were addressed, new features were engineered by transforming structural information within the existing data frames (e.g., flags for infinite fold-change, promoter orientation, chromosomal position, etc.).

### Feature Selection  
Models were built both on the complete feature set and on a filtered version of the dataset where highly collinear variables were removed (based on variance inflation factor thresholds).

### Model Building  
All models were implemented using a **`tidymodels`** workflow in R, executed on South Dakota State University’s HPC cluster. We trained multiple algorithms—including logistic regression, Multivariate Adaptive Regression Splines (MARS), Random Forest (RF), Recursive Partitioning (RPART), and Extreme Gradient Boosting (XGB)—as well as an ensemble combining their predictions.

### Model Evaluation  
For each algorithm and feature-set configuration, we performed 20 independent train/test splits. For every iteration, we saved both:
1. Train‐set and holdout‐set predictions, and  
2. Feature‐importance metrics (where applicable).  

These results were then aggregated to report mean performance metrics (e.g., ROC AUC) and to rank features by average importance across iterations.

---

## Results

### ROC AUC Performance  
(Receiver Operator Characteristic – Area Under Curve; 1.0 is perfect, 0.5 is random chance; appropriate for unbalanced classes like FOX genes.)

| Type  | Split | Ensemble | Logistic Regression | MARS | RF   | RPART | XGB  |
|-------|-------|----------|---------------------|------|------|-------|------|
| Test  | 0.7   | 0.80     | 0.74                | 0.75 | 0.80 | 0.72  | 0.78 |
| Test  | 0.8   | 0.80     | 0.74                | 0.75 | 0.80 | 0.70  | 0.79 |
| Test  | 0.9   | 0.82     | 0.72                | 0.77 | 0.82 | 0.76  | 0.82 |
| Train | 0.7   | 0.99     | 0.77                | 0.82 | 1.00 | 0.88  | 1.00 |
| Train | 0.8   | 0.99     | 0.77                | 0.81 | 1.00 | 0.84  | 1.00 |
| Train | 0.9   | 0.99     | 0.76                | 0.82 | 1.00 | 0.84  | 1.00 |

Model performance on the test set increases as more data is allocated to the training set. Each average ROC AUC value is calculated from 20 individual iterations with different train/test splits per model.





| Variable                                 | Estimate  | Std. Error | t value   | Pr(>|t|)   |
|------------------------------------------|-----------|------------|-----------|------------|
| `21 Hour Time Point Unique gene reads`   | 3.88E-05  | 1.51E-06   | 25.65664  | 1.27E-142  |
| `divergent_promoter_flag`                | 0.036585  | 0.002341   | 15.62603  | 1.10E-54   |
| `fold_12_inf`                            | 0.175083  | 0.01319    | 13.27354  | 5.08E-40   |
| `Chromosome region start`                | 1.26E-08  | 1.01E-09   | 12.53221  | 7.02E-36   |
| `0 Hour Time Point Unique gene reads`    | -2.63E-05 | 2.85E-06   | -9.20381  | 3.82E-20   |

Predictive variables for FOX gene classification and their biological relevance:

- **21 Hour Time Point Unique gene reads**: Higher unique read counts at 21 h post–nitrogen removal indicate genes strongly induced under nitrogen‐fixing conditions and are positively associated with FOX status.
- **divergent_promoter_flag**: Captures genes whose promoters are oriented oppositely to a neighboring gene—reflecting potential regulatory complexity that may favor a nitrogen‐fixation role.
- **fold_12_inf**: A flag for transcription fold‐change being infinite at 12 h (i.e., undetected at 0 h but highly expressed at 12 h), highlighting late‐induced FOX candidates.
- **Chromosome region start**: Chromosomal position may reflect operon context or genomic clustering of FOX gene families.
- **0 Hour Time Point Unique gene reads**: Negative coefficient suggests true FOX genes are lowly expressed at baseline (0 h) before induction.



---

```markdown
