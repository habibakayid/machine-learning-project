# Final optimization and stabilization

Executed 124 controlled candidate configurations across two separately fitted suites. Earlier models and results were preserved, and their hashes verified after fitting. Existing code, manifests, guidelines, studies and predictions were reviewed first.

## Primary Kaggle only

| Model | Source | Train accuracy / F1 | CV accuracy / F1 | Validation accuracy / F1 |
|---|---|---:|---:|---:|
| K-Means | kaggle | 50.98% / 49.82% | 49.29% / 48.87% | 53.49% / 53.22% |
| KNN Classifier | kaggle | 85.68% / 85.58% | 77.00% / 76.82% | 77.52% / 77.09% |
| Logistic Regression | kaggle | 94.03% / 94.01% | 77.49% / 77.35% | 76.90% / 76.55% |

## Supplemental mixed-source experiment

| Model | Source | Train accuracy / F1 | CV accuracy / F1 | Validation accuracy / F1 |
|---|---|---:|---:|---:|
| K-Means | field | 27.33% / 22.19% | 28.22% / 26.28% | 32.14% / 28.91% |
| K-Means | kaggle | 51.28% / 51.49% | 49.49% / 49.25% | 51.78% / 51.97% |
| KNN Classifier | field | 100.00% / 100.00% | 62.89% / 61.86% | 63.39% / 62.47% |
| KNN Classifier | kaggle | 100.00% / 100.00% | 72.72% / 72.59% | 71.01% / 70.83% |
| Logistic Regression | field | 90.89% / 90.96% | 62.67% / 62.00% | 55.36% / 55.93% |
| Logistic Regression | kaggle | 95.06% / 95.06% | 77.46% / 77.34% | 75.97% / 75.75% |


CV metrics are three-fold means. CV class metrics and confusion matrices pool one held-out prediction per original training image. Correct counts use that pooled denominator. The CSV includes pooled accuracy alongside mean fold accuracy.

## Model changes and evidence

LR stages keep gamma at 0.18 while varying C and landmarks. Matched ablations check HOG, LBP, edges, local descriptors and foreground cues. Supplemental weighting uses the same geometry. KNN varies neighborhoods in its main sweep, then compares fixed-neighborhood feature and PCA alternatives. K-Means reuses local features for the limited K=5/10/15 comparison. [Exact selections](FINAL_MODEL_SELECTION.md) · [Gaps and diagnostics](OVERFITTING_ANALYSIS.md)

For primary LR, removing HOG raised CV macro F1 to 77.53%, but validation F1 fell to 75.63% and failed the promotion check. The selected full-feature model gives 77.35% CV and 76.55% validation F1. Compact foreground features give 74.23% CV F1. Smaller landmark and simpler feature models remain in the comparison CSV; they did not beat the full selected model under the checks.

For primary KNN, at the same chosen neighborhood, PCA128 gives 66.18% CV F1 versus 76.82% for the selected Nyström/LDA representation. Removing HOG or local descriptors also lowered CV F1. The selected LDA representation uses labels strictly from the fitting fold.

For supplemental LR, source balancing at C=5 gives 69.61% joint CV F1 versus 69.67% with equal per-image weights. Its field validation accuracy is 56.25%, but its field gap is 43.75 pp and its Kaggle validation F1 is lower. Equal per-image weights remain selected. [Stage leaders](EXPERIMENT_STAGE_SUMMARY.csv) retain the best CV result in every stage, including rejected candidates and their rejection reasons.

## Weak classes and largest validation errors

| Suite | Model | Source | Lowest validation F1 | F1 | Largest error | Count |
|---|---|---|---|---:|---|---:|
| primary | K-Means | kaggle | rose | 37.45% | tulip → rose | 36 |
| primary | KNN Classifier | kaggle | daisy | 71.43% | daisy → dandelion | 21 |
| primary | Logistic Regression | kaggle | daisy | 71.86% | daisy → dandelion | 18 |
| supplemental | K-Means | field | tulip | 13.33% | tulip → dandelion | 17 |
| supplemental | K-Means | kaggle | rose | 39.73% | tulip → rose | 46 |
| supplemental | KNN Classifier | field | tulip | 48.78% | tulip → dandelion | 9 |
| supplemental | KNN Classifier | kaggle | daisy | 66.67% | rose → tulip | 21 |
| supplemental | Logistic Regression | field | tulip | 35.56% | tulip → dandelion | 11 |
| supplemental | Logistic Regression | kaggle | daisy | 70.27% | daisy → dandelion | 19 |

## K-Means clustering diagnostics

Silhouette, ARI and NMI describe raw cluster assignments, before mapping to flower names. CV rows below average the three separately fitted folds. Partitions above 800 rows use a seeded 800-row silhouette sample; other partitions use every row.

| Suite | Source | Evaluation | Clusters | Silhouette | ARI | NMI |
|---|---|---|---:|---:|---:|---:|
| primary | kaggle | cv | 10 | 0.0845 | 0.1104 | 0.1782 |
| primary | kaggle | train | 10 | 0.0774 | 0.1060 | 0.1667 |
| primary | kaggle | validation | 10 | 0.0786 | 0.1169 | 0.1960 |
| supplemental | field | cv | 15 | 0.0479 | 0.0377 | 0.1777 |
| supplemental | field | train | 15 | 0.0489 | 0.0252 | 0.1087 |
| supplemental | field | validation | 15 | 0.0009 | 0.0490 | 0.2043 |
| supplemental | kaggle | cv | 15 | 0.0827 | 0.0980 | 0.1776 |
| supplemental | kaggle | train | 15 | 0.0783 | 0.0959 | 0.1827 |
| supplemental | kaggle | validation | 15 | 0.0846 | 0.0955 | 0.2026 |

Image similarity does not necessarily follow flower taxonomy. Background, lighting and color can dominate Euclidean clusters. Additional clusters allow several appearances per flower class, but weak silhouette and low field ARI show that this remains a limited classifier. Labels name clusters only after centroid fitting; they do not guide K-Means.

## Error changes against each suite's baseline

The [error-change table](ERROR_CHANGES.csv) records every validation count before and after selection. Positive error_reduction means fewer mistakes for that true/predicted pair. Diagonal changes describe correct predictions and should be interpreted separately.

| Suite | Model | Source | Largest corrected confusion | Fewer errors |
|---|---|---|---|---:|
| primary | K-Means | kaggle | dandelion → daisy | 46 |
| primary | KNN Classifier | kaggle | tulip → rose | 17 |
| primary | Logistic Regression | kaggle | daisy → rose | 3 |
| supplemental | K-Means | field | daisy → dandelion | 0 |
| supplemental | K-Means | kaggle | daisy → dandelion | 0 |
| supplemental | KNN Classifier | field | rose → daisy | 2 |
| supplemental | KNN Classifier | kaggle | dandelion → tulip | 2 |
| supplemental | Logistic Regression | field | daisy → dandelion | 1 |
| supplemental | Logistic Regression | kaggle | tulip → rose | 2 |

Similar colors, petal structure and framing are plausible causes of confusion. Earlier contact sheets showed wild roses among foliage and slender yellow field tulips. These are qualitative findings, without segmentation or species ground truth for Kaggle. [Earlier framing audit](../generalization/framing_summary.csv)

## Run and verification

From the folder containing gui and part_b_images, run the command below. The Training data selector defaults to Kaggle only. The supplemental selector labels mixed-source training.

    .\.venv\Scripts\python.exe -m streamlit run gui/image_app.py

Saved bundles were checked against stored predictions and five direct image predictions each. [Browser checks](GUI_VERIFICATION.md) record automated image-upload verification.

## Submission status and limitations

The primary fitting protocol now uses one dataset. The course restriction on AI-written code needs resolution with the TA. This workspace has empty Part A and Part C folders and no Git history. Part B technical readiness does not establish whole-course submission eligibility.

CV and validation images have informed repeated development decisions. They are development estimates. No new common external test was fabricated, and historical tests remain cohort-specific. Frozen-representation LOO and group-excluded neighbor scores are descriptive only. Exact end-to-end leave-one-out refitting was not run. No presentation PDF was created.

## Source artifacts

[All experiments](ALL_EXPERIMENTS.csv) · [Final metrics](FINAL_MODEL_METRICS.csv) · [All candidate per-class metrics](ALL_PER_CLASS_METRICS.csv) · [Final per-class metrics](FINAL_PER_CLASS_METRICS.csv) · [All candidate confusion counts](ALL_CANDIDATE_CONFUSIONS.csv) · [Final confusion counts](FINAL_CONFUSION_MATRICES.csv) · [Model manifest](MODEL_MANIFEST.json) · [Figures](figures/)
