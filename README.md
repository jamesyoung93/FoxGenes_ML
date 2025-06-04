# FoxGenes_ML

Data Sources / References:

1. Lechno-Yossef et al., 2011
Title: Identification of ten Anabaena sp. genes that under aerobic conditions are required for growth on dinitrogen but not for growth on fixed nitrogen
Journal: Journal of Bacteriology

Supplementary Table of this work provides about 75 experimentally validated FOX genes.
Established ~10 FOX genes via insertional mutagenesis.


2. Tiruveedhula & Wangikar, 2017
Title: Gene essentiality, conservation index and co-evolution of genes in cyanobacteria
Journal: PLOS One

Findings of non-essential genes, many conserved.
Introduced the concept of NotFOX genes so the model could have positive and negative examples.

3. Flaherty et al., 2011
Title: Directional RNA deep sequencing sheds new light on the transcriptional response of Anabaena sp. strain PCC 7120 to combined-nitrogen deprivation
Journal: BMC Genomics

Time-series RNA-Seq under N-deprivation: 0h, 6h, 12h, 21h.
Key input for transcriptomic-based FOX prediction.

4. Zhang et al., 2021
Title: Quantitative Proteomics Reveals the Protein Regulatory Network of Anabaena sp. PCC 7120 under Nitrogen Deficiency
Journal: Journal of Proteome Research

Provided protein fold-changes and p-values under N-starvation.

Methods
Data Gathering 
Data from prior transcriptomic and proteomic experiments on Nostoc 7120 were gathered from the sources above. These data frames were joined based on gene names present in both datasets. This bioinformatic data was further joined with experimentally validated FOX genes from 3. The remaining genes were either marked as “unknown” (may or may not be a FOX gene) or “NotFOX”. Determination of “NotFOX” genes was made from a study of non-essential genes that are conserved across cyanobacteria in general (Tiruveedula, 2017 #449).  

Data Cleaning and Feature Engineering
Some genes were missing information from transcriptomic and proteomics data. This also lead to missing information that was derivative of these variables, such as fold change. These situations of missing data were handled by creating a binned representation of the data as well as filling in zeroes for the missing data in it’s original column. No genes were deleted from the dataset due to missing data. Once missing data was handled, new variables were created by transforming the structural information within the existing data.

Feature Selection
Models were built on both the complete feature set and a filtered version of the dataset the removed variables based on collinearity. 

Model Building
Models were built using a tidymodel workflow in the R language on South Dakota State University’s HPC resources.  

Model Evaluation
The 20 iterations of each model/feature set/optimization combination had their individual train and holdout set predictions saved as well as the feature importance in the final model of each iteration. These were used for feature ranking based on mean importance. 

Results
ROC AUC (Receiver Operator Characteristic - Area Under Curve, 1 is perfect, 0.5 is random chance, this is used for unbalanced data, which FOX genes are due to their rarity)

| Type  | Split | Ensemble | Logistic Regression | MARS | RF   | RPART | XGB  |
|-------|-------|----------|---------------------|------|------|-------|------|
| Test  | 0.7   | 0.80     | 0.74                | 0.75 | 0.80 | 0.72  | 0.78 |
| Test  | 0.8   | 0.80     | 0.74                | 0.75 | 0.80 | 0.70  | 0.79 |
| Test  | 0.9   | 0.82     | 0.72                | 0.77 | 0.82 | 0.76  | 0.82 |
| Train | 0.7   | 0.99     | 0.77                | 0.82 | 1.00 | 0.88  | 1.00 |
| Train | 0.8   | 0.99     | 0.77                | 0.81 | 1.00 | 0.84  | 1.00 |
| Train | 0.9   | 0.99     | 0.76                | 0.82 | 1.00 | 0.84  | 1.00 |

Model performance on the test set increases as more data is allocated to the train set. Each average ROC_AUC is calculated from the 20 individual iterations with different train/test datasets within each model.

Feature Importance from simple and interpretable logistic regression (Top 5)

| Variable                                 | Estimate  | Std. Error | t value   | Pr(>|t|)     |
|------------------------------------------|-----------|------------|-----------|--------------|
| `21 Hour Time Point Unique gene reads`   | 3.88E-05  | 1.51E-06   | 25.65664  | 1.27E-142    |
| divergent_promoter_flag                  | 0.036585  | 0.002341   | 15.62603  | 1.10E-54     |
| fold_12_inf                              | 0.175083  | 0.01319    | 13.27354  | 5.08E-40     |
| `Chromosome region start`                | 1.26E-08  | 1.01E-09   | 12.53221  | 7.02E-36     |
| `0 Hour Time Point Unique gene reads`    | -2.63E-05 | 2.85E-06   | -9.20381  | 3.82E-20     |

Predictive variables for FOX gene classification and their biological relevance. Higher unique read counts at 21 hours post–nitrogen removal indicate genes strongly induced under nitrogen‐fixing conditions and are positively associated with FOX status, whereas higher baseline (0 hour) expression is negatively associated (suggesting true FOX genes are lowly expressed before induction). The divergent_promoter_flag captures genes with promoters oriented opposite a neighbor, reflecting potential regulatory complexity that favors nitrogen‐fixation function. The fold_12_inf flag denotes genes with infinite fold‐change at 12 hours (i.e., undetected at 0 hours but highly expressed at 12 hours), highlighting late‐induced candidates. Finally, chromosome region start position may reflect operon or genomic context important for FOX gene clusters.









