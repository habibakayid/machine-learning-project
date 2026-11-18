# Development evaluation

The tables use the same fixed manifests. Training and validation are newly checked with saved artifacts. CV metrics are means of three fold-level scores reproduced from saved out-of-fold predictions. Correct counts, per-class scores, and CV confusion matrices pool the held-out predictions once per image. These pooled counts can differ slightly from mean fold accuracy. A dash means the metric is unavailable. The Kaggle-only models have no fitting accuracy on field photos.

[Development metrics](FINAL_METRICS.csv) · [Per-class precision, recall and F1](FINAL_PER_CLASS.csv) · [Confusion counts and percentages](FINAL_CONFUSIONS.csv)

## Kaggle Training

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | 2996/3017 | 99.30% | 99.28% | 99.29% | 99.28% | new development inference |
| Field-trained Nyström LR (final) | 2975/3017 | 98.61% | 98.60% | 98.64% | 98.61% | new development inference |
| Regularized balanced C=3 LR | 2662/3017 | 88.23% | 88.05% | 88.52% | 88.25% | new development inference |
| Compact LR | 2322/3017 | 76.96% | 76.70% | 77.17% | 76.86% | new development inference |
| Rebuilt KNN | 3017/3017 | 100.00% | 100.00% | 100.00% | 100.00% | new development inference |
| Compact field KNN | 2164/3017 | 71.73% | 71.66% | 72.01% | 71.59% | new development inference |
| Nyström + LDA KNN (final) | 3017/3017 | 100.00% | 100.00% | 100.00% | 100.00% | new development inference |
| Rebuilt K=5 K-Means | 1173/3017 | 38.88% | 38.70% | 39.35% | 38.63% | new development inference |
| Compact K=5 K-Means | 1196/3017 | 39.64% | 39.18% | 40.19% | 38.10% | new development inference |
| Local words K=15 K-Means (final) | 1547/3017 | 51.28% | 54.56% | 50.85% | 51.49% | new development inference |

## Kaggle Grouped cross-validation

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | 2320/3017 | 76.90% | 76.77% | 76.66% | 76.66% | saved out-of-fold predictions |
| Field-trained Nyström LR (final) | 2313/3017 | 76.67% | 76.50% | 76.62% | 76.51% | saved out-of-fold predictions |
| Regularized balanced C=3 LR | 2279/3017 | 75.54% | 75.27% | 75.80% | 75.37% | saved out-of-fold predictions |
| Compact LR | 2091/3017 | 69.31% | 69.16% | 69.47% | 69.21% | saved out-of-fold predictions |
| Rebuilt KNN | 2026/3017 | 67.15% | 67.05% | 67.18% | 66.96% | saved out-of-fold predictions |
| Compact field KNN | 1944/3017 | 64.44% | 64.48% | 64.88% | 64.24% | saved out-of-fold predictions |
| Nyström + LDA KNN (final) | 2178/3017 | 72.19% | 72.09% | 72.15% | 72.05% | saved out-of-fold predictions |
| Rebuilt K=5 K-Means | 1179/3017 | 39.08% | 38.92% | 39.56% | 38.85% | saved out-of-fold predictions |
| Compact K=5 K-Means | 1123/3017 | 37.22% | 35.78% | 37.29% | 35.44% | saved out-of-fold predictions |
| Local words K=15 K-Means (final) | 1493/3017 | 49.49% | 52.20% | 49.41% | 49.25% | saved out-of-fold predictions |

## Kaggle Validation

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | 493/645 | 76.43% | 76.28% | 76.44% | 76.21% | new development inference |
| Field-trained Nyström LR (final) | 488/645 | 75.66% | 75.57% | 75.50% | 75.48% | new development inference |
| Regularized balanced C=3 LR | 478/645 | 74.11% | 73.77% | 74.20% | 73.83% | new development inference |
| Compact LR | 444/645 | 68.84% | 68.86% | 68.60% | 68.36% | new development inference |
| Rebuilt KNN | 425/645 | 65.89% | 65.80% | 66.09% | 65.86% | new development inference |
| Compact field KNN | 416/645 | 64.50% | 64.32% | 64.76% | 64.28% | new development inference |
| Nyström + LDA KNN (final) | 460/645 | 71.32% | 71.47% | 71.33% | 71.30% | new development inference |
| Rebuilt K=5 K-Means | 236/645 | 36.59% | 36.50% | 37.31% | 36.39% | new development inference |
| Compact K=5 K-Means | 239/645 | 37.05% | 36.71% | 37.28% | 35.57% | new development inference |
| Local words K=15 K-Means (final) | 334/645 | 51.78% | 54.79% | 51.27% | 51.97% | new development inference |

## Field Training

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | — | — | — | — | — | Unavailable |
| Field-trained Nyström LR (final) | 438/450 | 97.33% | 97.36% | 97.31% | 97.31% | new development inference |
| Regularized balanced C=3 LR | 446/450 | 99.11% | 99.00% | 99.17% | 99.07% | new development inference |
| Compact LR | 408/450 | 90.67% | 91.37% | 90.68% | 90.83% | new development inference |
| Rebuilt KNN | — | — | — | — | — | Unavailable |
| Compact field KNN | 243/450 | 54.00% | 58.13% | 53.84% | 53.20% | new development inference |
| Nyström + LDA KNN (final) | 450/450 | 100.00% | 100.00% | 100.00% | 100.00% | new development inference |
| Rebuilt K=5 K-Means | — | — | — | — | — | Unavailable |
| Compact K=5 K-Means | 113/450 | 25.11% | 29.93% | 25.34% | 19.35% | new development inference |
| Local words K=15 K-Means (final) | 123/450 | 27.33% | 32.00% | 26.29% | 22.19% | new development inference |

## Field Grouped cross-validation

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | 209/450 | 46.44% | 52.07% | 46.76% | 43.75% | saved out-of-fold predictions |
| Field-trained Nyström LR (final) | 275/450 | 61.11% | 62.54% | 60.80% | 60.52% | saved out-of-fold predictions |
| Regularized balanced C=3 LR | 296/450 | 65.78% | 67.63% | 65.27% | 65.12% | saved out-of-fold predictions |
| Compact LR | 285/450 | 63.33% | 65.42% | 62.59% | 62.68% | saved out-of-fold predictions |
| Rebuilt KNN | — | — | — | — | — | Unavailable |
| Compact field KNN | 198/450 | 44.00% | 47.54% | 43.62% | 42.10% | saved out-of-fold predictions |
| Nyström + LDA KNN (final) | 285/450 | 63.33% | 63.62% | 62.50% | 61.96% | saved out-of-fold predictions |
| Rebuilt K=5 K-Means | — | — | — | — | — | Unavailable |
| Compact K=5 K-Means | 110/450 | 24.44% | 30.00% | 24.24% | 20.18% | saved out-of-fold predictions |
| Local words K=15 K-Means (final) | 127/450 | 28.22% | 39.06% | 27.69% | 26.28% | saved out-of-fold predictions |

## Field Validation

| Model | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Original rebuilt Nyström LR | 51/112 | 45.54% | 48.33% | 45.83% | 43.50% | new development inference |
| Field-trained Nyström LR (final) | 61/112 | 54.46% | 54.81% | 55.42% | 54.92% | new development inference |
| Regularized balanced C=3 LR | 61/112 | 54.46% | 54.41% | 55.42% | 54.67% | new development inference |
| Compact LR | 68/112 | 60.71% | 62.35% | 60.83% | 61.43% | new development inference |
| Rebuilt KNN | 42/112 | 37.50% | 36.02% | 39.17% | 34.50% | new development inference |
| Compact field KNN | 55/112 | 49.11% | 48.97% | 50.42% | 47.62% | new development inference |
| Nyström + LDA KNN (final) | 70/112 | 62.50% | 64.22% | 61.67% | 62.04% | new development inference |
| Rebuilt K=5 K-Means | 23/112 | 20.54% | 24.44% | 21.25% | 18.80% | new development inference |
| Compact K=5 K-Means | 25/112 | 22.32% | 23.30% | 23.33% | 18.08% | new development inference |
| Local words K=15 K-Means (final) | 36/112 | 32.14% | 37.34% | 31.25% | 28.91% | new development inference |
