# Final optimization results

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
