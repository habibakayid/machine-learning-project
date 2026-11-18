# Overfitting analysis

| Suite | Model | Source | Train accuracy | Validation accuracy | Gap |
|---|---|---|---:|---:|---:|
| primary | K-Means | kaggle | 50.98% | 53.49% | -2.51 pp |
| primary | KNN Classifier | kaggle | 85.68% | 77.52% | 8.16 pp |
| primary | Logistic Regression | kaggle | 94.03% | 76.90% | 17.13 pp |
| supplemental | K-Means | field | 27.33% | 32.14% | -4.81 pp |
| supplemental | K-Means | kaggle | 51.28% | 51.78% | -0.51 pp |
| supplemental | KNN Classifier | field | 100.00% | 63.39% | 36.61 pp |
| supplemental | KNN Classifier | kaggle | 100.00% | 71.01% | 28.99 pp |
| supplemental | Logistic Regression | field | 90.89% | 55.36% | 35.53 pp |
| supplemental | Logistic Regression | kaggle | 95.06% | 75.97% | 19.09 pp |

## KNN diagnostics

Resubstitution includes each query as a possible zero-distance neighbor. The leave-one-out and same-group-excluded neighbor diagnostics freeze the learned representation. They omit the stated neighbors, but the scaler, vocabulary, Nyström map and LDA have seen those training images. They are descriptive fitting diagnostics.

The group_excluded_refitted_3fold row excludes the held-out groups from every transformation and the neighbor pool. It is computed from the fold-refitted training predictions and equals grouped CV. It fits about two thirds of groups per fold. Exact end-to-end leave-one-image-out or leave-one-group-out refitting was not executed. Frozen-representation LOO is explicitly distinguished from those independent-refit estimates.

| Suite | Source | Evaluation | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| primary | kaggle | cv | 77.00% | 76.82% |
| primary | kaggle | group_excluded_frozen_representation | 84.82% | 84.73% |
| primary | kaggle | group_excluded_refitted_3fold | 77.00% | 76.82% |
| primary | kaggle | leave_one_out_frozen_representation | 84.82% | 84.73% |
| primary | kaggle | train | 85.68% | 85.58% |
| primary | kaggle | validation | 77.52% | 77.09% |
| supplemental | field | cv | 62.89% | 61.86% |
| supplemental | field | group_excluded_frozen_representation | 90.22% | 90.00% |
| supplemental | field | group_excluded_refitted_3fold | 62.89% | 61.86% |
| supplemental | field | leave_one_out_frozen_representation | 90.22% | 90.00% |
| supplemental | field | train | 100.00% | 100.00% |
| supplemental | field | validation | 63.39% | 62.47% |
| supplemental | kaggle | cv | 72.72% | 72.59% |
| supplemental | kaggle | group_excluded_frozen_representation | 78.49% | 78.34% |
| supplemental | kaggle | group_excluded_refitted_3fold | 72.72% | 72.59% |
| supplemental | kaggle | leave_one_out_frozen_representation | 78.52% | 78.37% |
| supplemental | kaggle | train | 100.00% | 100.00% |
| supplemental | kaggle | validation | 71.01% | 70.83% |

LR gaps are assessed together with CV, validation and class scores. Lower training accuracy alone does not select a model. K-Means uses labels only to name clusters after fitting. Low training and validation scores together indicate weak alignment of image clusters and flower labels. Silhouette uses Euclidean coordinates and raw clusters in the fitted space. ARI and NMI use raw cluster IDs and the correct partition labels. Large partitions use a seeded 800-row silhouette sample.
