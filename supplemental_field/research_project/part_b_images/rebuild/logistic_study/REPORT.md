# Logistic Regression regularization study

The study reused the existing five classes and handcrafted features, including the same center crop and feature group scaling for the matched Nyström comparisons. It kept 3,017 Kaggle and 450 supplemental field training photos, 645 Kaggle and 112 field validation photos, and the original three grouped CV folds. The folds keep related Kaggle images and each field photographer together. The codebook, the group scalers, and the Nyström transform were fitted inside each fold. [Fixed development assignments](../generalization/development_split.csv)

Selection uses the mean of Kaggle and field CV macro F1, with each source given equal importance. The search first tested all seven C values at the original 1,300 landmarks and gamma 0.18. It carried the strongest C into the landmark comparison, then tested lower gamma values around the winning C to account for changes in feature scale. The feature ablations used the leading geometry. The weighting comparison retained every training photo and changed the Logistic Regression fitting weights. Source balancing gives each source equal total weight. Source-and-class balancing also gives each class equal weight within each source. A Streamlit change requires a higher selection score and smaller validation gaps for both sources.

The saved current model gives every fitting photo equal weight in its loss. The compact benchmark uses the pre-existing source-balanced fit. The 645-image internal test and the previous 234-, 150-, and 191-photo field tests were excluded from candidate selection and fitting. [Candidate CSV](experiments.csv) · [Prior generalization report](../generalization/REPORT.md) · [Nyström parameters](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html)

The fresh test contains 156 iNaturalist photos from 2016–2019. The sampling blocked observers in training, development validation, and the three previous field tests. The overlap audit checked file and decoded-pixel hashes, photo IDs where available, and likely perceptual copies against the full raw Kaggle collection and prior field collections. The fixed manifest contains 36 daisy, 36 dandelion, 36 rose, 25 sunflower, and 23 tulip photos. It was prepared independently of prediction scores. [Manifest, attributions, and licenses](fresh_manifest.csv) · [Overlap audit exclusions](overlap_exclusions.csv)

## Staged development results

| C at 1,300 landmarks, gamma 0.18, unweighted | Kaggle CV macro F1 | Field CV macro F1 | Equal-source mean |
|---:|---:|---:|---:|
| 0.03 | 66.94% | 31.89% | 49.42% |
| 0.10 | 72.05% | 43.63% | 57.84% |
| 0.30 | 75.12% | 51.79% | 63.45% |
| 1 | 76.84% | 57.27% | 67.06% |
| 3 | 77.51% | 61.12% | **69.31%** |
| 10 | 76.71% | 60.89% | 68.80% |
| 18 | 76.51% | 60.52% | 68.52% |

- **Landmarks.** At C=3, the balanced CV macro F1 was 65.00%, 66.49%, 67.66%, and 69.31% for 300, 600, 900, and 1,300 landmarks. The 600-landmark model met the Kaggle gap goal at 12.13 points, with a field gap of 27.33 points and a lower CV score than the current model.
- **Gamma.** At 1,300 landmarks, the best score with gamma 0.03 was 67.51% and with gamma 0.08 it was 68.45%, both at C=10. Gamma 0.18 retained the best result.
- **Feature groups.** Removing HOG scored 68.47% at C=3. Removing the local visual words scored 65.71%. Keeping only color, texture, and edges scored 66.28%. The full representation retained the best result.
- **Weights.** Source balancing raised the C=3 score to 69.90%. Source-and-class balancing raised it to 70.25%, using all training photos. Both methods increased the influence of field photos during fitting. The CV scoring rule stayed the same. At C=1, source balancing and source-and-class balancing scored 68.81% and 68.55%, respectively.

The selected configuration uses C=3, degree 2, gamma 0.18, 1,300 landmarks, the full feature set, and source-and-class balancing. All 24 experiments include training, CV, validation, and gap metrics by source. [All experiments](experiments.csv) · [Selected configuration](selected.csv) · [Saved candidate](regularized_logistic.joblib)

### Accuracy and macro F1

Each cell shows **accuracy / macro F1**, in percent. Gaps are training accuracy minus validation accuracy.

| Source and split | Current Nyström | Selected C=3 | Compact benchmark |
|---|---:|---:|---:|
| Kaggle train | 98.61 / 98.61 | 88.23 / 88.25 | 76.96 / 76.86 |
| Kaggle CV | 76.67 / 76.51 | 75.54 / 75.37 | 69.31 / 69.21 |
| Kaggle validation | 75.66 / 75.48 | 74.11 / 73.83 | 68.84 / 68.36 |
| Kaggle gap | 22.95 pp | 14.12 pp | 8.13 pp |
| Field train | 97.33 / 97.31 | 99.11 / 99.07 | 90.67 / 90.83 |
| Field CV | 61.11 / 60.52 | 65.78 / 65.12 | 63.33 / 62.68 |
| Field validation | 54.46 / 54.92 | 54.46 / 54.67 | 60.71 / 61.43 |
| Field gap | 42.87 pp | 44.65 pp | 29.95 pp |
| Mean CV macro F1 | 68.52% | **70.25%** | 65.95% |

The winner reached the Kaggle gap goal. Its field validation accuracy stayed at 54.46% as its field training accuracy increased, so that gap grew. The unweighted C=3 variant reduced the gaps to 17.24 and 37.33 points with 69.31% balanced CV macro F1. Source balancing at C=1 reduced both gaps to 10.58 and 40.20 points with 68.81% balanced CV macro F1. The CV winner did not meet the two-gap deployment criterion. Streamlit continues to use the existing Logistic Regression bundle. [Validation confusion matrices](validation_confusions.png) · [Grouped CV confusion matrices](cv_confusions.png) · [Development per-class recall and F1](development_per_class.csv) · [Counts and row percentages](development_confusions.csv)

## New independent test

After selection, the three saved models were evaluated on the same 156 photos.

| Model | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Current Nyström | 100 | 64.10% | 62.66% |
| **Selected C=3** | **108** | **69.23%** | **67.39%** |
| Compact benchmark | 102 | 65.38% | 64.41% |

| Flower | Support | Current recall | Selected recall | Selected F1 |
|---|---:|---:|---:|---:|
| Daisy | 36 | 63.89% | 75.00% | 75.00% |
| Dandelion | 36 | 75.00% | 63.89% | 64.79% |
| Rose | 36 | 72.22% | 77.78% | 77.78% |
| Sunflower | 25 | 68.00% | 88.00% | 83.02% |
| Tulip | 23 | 30.43% | 34.78% | 36.36% |

The selected model improved by 5.13 accuracy points and 4.73 macro F1 points over the current one on the new sample. Sunflower had the largest gain in recall. Dandelion recall fell, and tulip remained the weakest class. The historical 74.67% accuracy came from a separate 150-image cohort. [Fresh per-class metrics for all models](fresh_per_class.csv) · [Fresh confusion matrices](fresh_confusions.png) · [Fresh predictions](fresh_predictions.csv) · [Run commands](../../../README.md)
