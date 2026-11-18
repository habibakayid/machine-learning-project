# Baseline captured before the optimization runs

The existing no-flags demo used supplemental mixed-source models. Its saved metrics
were copied to [BASELINE_COMPARISON.csv](BASELINE_COMPARISON.csv) before any fitting
in this study. SHA-256 hashes of every earlier rebuilt/study model were written
to [baseline_snapshot.json](baseline_snapshot.json). The earlier bundles remain
the historical reference.

| Model | Kaggle train | Kaggle CV | Kaggle validation | Field train | Field CV | Field validation |
|---|---:|---:|---:|---:|---:|---:|
| Nyström LR, C=18, 1,300 landmarks | 98.61% | 76.67% | 75.66% | 97.33% | 61.11% | 54.46% |
| Nyström/LDA Manhattan KNN, k=15 | 100.00% | 72.19% | 71.32% | 100.00% | 63.33% | 62.50% |
| Local words/PCA20 K-Means, k=15 | 51.28% | 49.49% | 51.78% | 27.33% | 28.22% | 32.14% |

The primary Kaggle-only baselines are the archived rebuild bundles. Their CV
accuracies are 76.90%, 67.15%, and 39.08% for LR, KNN, and K-Means, respectively.
Their validation accuracies are 76.43%, 65.89%, and 36.59%. The current study refits
the same recipes on the saved partitions before comparing new candidates.

The source PDF requires one dataset per project part. Primary fitting in this
study uses the original Kaggle images only. Supplemental training is an explicitly
separate generalization experiment. No documented instructor approval for combining
the sources was found in the workspace.

The prior primary and field test cohorts have all been accessed. The study runner
loads only the development manifests. It selects models using the predeclared
[development policy](SELECTION_POLICY.md). A common unseen test for the three
newly selected primary models is unavailable.
