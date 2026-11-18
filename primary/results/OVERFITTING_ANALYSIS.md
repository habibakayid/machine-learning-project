# Primary overfitting diagnostics

| Model | Train accuracy | Validation accuracy | Gap (pp) |
|---|---:|---:|---:|
| K-Means | 50.98% | 53.49% | -2.51 |
| KNN Classifier | 85.68% | 77.52% | 8.16 |
| Logistic Regression | 94.03% | 76.90% | 17.13 |

KNN ordinary resubstitution permits the query to be its own neighbor. Frozen-representation LOO and same-group-neighbor exclusions are descriptive only because the representation has seen the fitting images. The independent group-excluded diagnostic is fold-refitted three-fold CV; no exact end-to-end LOO was executed.

The full original gap analysis and rejected configurations remain in supplemental_field/research_project/part_b_images/rebuild/final_optimization/OVERFITTING_ANALYSIS.md. This primary-only summary changes no stored results. No new learning curve has been trained.
