# Confusion matrix audit

Every matrix uses Daisy, Dandelion, Rose, Sunflower and Tulip in that order. Rows are true labels and columns are predicted labels. Each normalized cell is its raw count divided by the true class's support. Display rounding can make a rendered row sum differ slightly from 100%.

## Existing records

- Eight CSV files contain **403 prediction-backed confusion matrices**. The audit rebuilt all 10,075 cells from saved predictions and checked counts, percentages, row totals, and the diagonal. It checked test prediction paths and true labels against their own manifests.
- The three legacy 647-image PNGs were read as integer counts and cross-checked with archived class supports, accuracy, and per-class precision, recall, and F1. The legacy K-Means matrix also agreed with the stored cluster contingency table and the archived mapping. Legacy sample-level predictions are unavailable.
- The 18 pre-existing matrix PNG files were checked for readability, axis direction, title and class ordering. The earlier Logistic Regression study plots did not say CV or validation in their visible titles. The archived images were preserved and split-labeled copies were saved. The earlier generic test titles belong to the specific cohorts documented in [Test cohorts](TEST_COHORTS.md), so they should travel with that context.

[CSV verification manifest](CONFUSION_CHECKS.csv) · [Labeled Logistic CV copy](figures/logistic_study_cv_labeled.png) · [Labeled Logistic validation copy](figures/logistic_study_validation_labeled.png)

## Final development figures

Each figure includes training, pooled grouped CV, and validation, with raw counts beside row-normalized percentages.

| Final model | Kaggle | Field |
|---|---|---|
| Logistic Regression | [Kaggle](figures/lr_mixed_kaggle.png) | [Field](figures/lr_mixed_field.png) |
| KNN | [Kaggle](figures/knn_final_kaggle.png) | [Field](figures/knn_final_field.png) |
| K-Means | [Kaggle](figures/km_final_kaggle.png) | [Field](figures/km_final_field.png) |

The CV diagonal gives pooled correctness. The standardized CV accuracy column is a mean of the three fold accuracies, which can differ slightly. [Development confusion matrices](FINAL_CONFUSIONS.csv) · [Test confusion matrices, separate by cohort](TEST_CONFUSIONS.csv)

## Largest validation errors

| Model | Kaggle | Field |
|---|---|---|
| Logistic Regression | Daisy → Dandelion, 18 | Tulip → Dandelion, 10 |
| KNN | Rose → Tulip, 21 | Tulip → Dandelion, 8 |
| K-Means | Tulip → Rose, 46 | Tulip → Dandelion, 17 |

[Top three counts and row percentages per model and source](TOP_ERRORS.csv)
