# Test cohorts

All listed scores are historical. The audit recomputed metrics from their saved predictions or checked archived integer confusion counts. It made no new test predictions. The independent field cohorts were independent when scored and have since been accessed. The current audit found zero cross-cohort observer or detected image matches among the field cohorts and supplemental development partition.

The final K-Means model has no independent test predictions. The 27.33% result belongs to the archived compact K=5 model. There is no common independent test for all three deployed bundles. The 74.67% field-trained LR and 63.35% final KNN scores come from different cohorts. Use the paired 191-photo comparison to compare the two deployed supervised models.

[Test metrics](TEST_METRICS.csv) · [Per-class precision, recall and F1](TEST_PER_CLASS.csv) · [Counts and row percentages by cohort](TEST_CONFUSIONS.csv) · [Class counts](DATA_SPLITS.csv) · [Overlap checks](DATA_OVERLAP.csv)

## Legacy Kaggle test · 647 photos

Previously accessed split from the original scripts. Eight detected related-photo groups crossed the legacy train/validation/test boundaries. The integer confusion matrices were read from the archived PNGs and checked against the original summary and per-class scores.

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Legacy Logistic Regression | 371/647 | 57.34% | 57.34% | 56.76% | 56.92% | historical PNG counts checked against saved metrics |
| Legacy KNN | 302/647 | 46.68% | 47.76% | 45.67% | 45.59% | historical PNG counts checked against saved metrics |
| Legacy K-Means | 177/647 | 27.36% | 15.50% | 24.41% | 18.59% | historical PNG counts checked against saved metrics |

## Rebuilt Kaggle internal test · 645 photos

A held-out partition for the rebuilt pipeline using the same raw Kaggle collection. In the legacy split, these photos were assigned 452 to training, 94 to validation, and 99 to test. Results cover the original rebuilt models, not the three deployed bundles.

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | 482/645 | 74.73% | 74.56% | 74.68% | 74.58% | saved test predictions |
| Rebuilt KNN | 454/645 | 70.39% | 70.43% | 70.30% | 70.22% | saved test predictions |
| Rebuilt K=5 K-Means | 252/645 | 39.07% | 38.80% | 39.64% | 38.81% | saved test predictions |

## 2025 external field test · 234 photos

Independent when collected for the original rebuilt Nyström LR. These results had already been published before the final audit.

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | 109/234 | 46.58% | 49.08% | 44.35% | 42.37% | saved test predictions |

## 2026 external field test · 150 photos

Independent when collected for the generalization study. The field-trained and original Nyström, Compact, archived compact KNN, and archived compact K-Means were compared on this cohort.

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | 79/150 | 52.67% | 57.88% | 53.24% | 51.01% | saved test predictions |
| Field-trained Nyström LR (final) | 112/150 | 74.67% | 72.18% | 72.64% | 71.81% | saved test predictions |
| Compact LR | 107/150 | 71.33% | 70.22% | 70.01% | 69.16% | saved test predictions |
| Compact field KNN | 78/150 | 52.00% | 63.35% | 49.98% | 50.58% | saved test predictions |
| Compact K=5 K-Means | 41/150 | 27.33% | 29.39% | 23.77% | 18.06% | saved test predictions |

## 2020–2021 external field test · 191 photos

Independent when collected for the KNN study. This is the directly comparable external test for the two deployed supervised models, both scored on these same 191 images.

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Field-trained Nyström LR (final) | 119/191 | 62.30% | 63.68% | 62.43% | 62.05% | saved test predictions |
| Compact field KNN | 82/191 | 42.93% | 44.66% | 42.58% | 40.05% | saved test predictions |
| Nyström + LDA KNN (final) | 121/191 | 63.35% | 65.03% | 62.96% | 62.93% | saved test predictions |

## 2016–2019 external field test · 156 photos

Independent when collected for the Logistic Regression regularization study. This compares the field-trained, regularized C=3, and Compact variants on the same images.

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Field-trained Nyström LR (final) | 100/156 | 64.10% | 64.72% | 61.91% | 62.66% | saved test predictions |
| Regularized balanced C=3 LR | 108/156 | 69.23% | 67.03% | 67.89% | 67.39% | saved test predictions |
| Compact LR | 102/156 | 65.38% | 65.18% | 64.21% | 64.41% | saved test predictions |
