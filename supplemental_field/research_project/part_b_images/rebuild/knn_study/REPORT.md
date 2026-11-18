# KNN follow-up

## Baseline and experiment design

The archived KNN standardized four compact color and shape groups, scaled them by the configured group weight divided by the square root of group size, and kept 64 PCA components. It used 15 Euclidean neighbors with uniform voting. Its training pool contained 3,017 Kaggle photos and 450 supplemental field photos, so field photos made up 12.98% of the pool. PCA retained directions with high overall variance. It had no access to class labels when choosing those directions. The published 2026 test showed low recall for rose and sunflower, plus frequent dandelion predictions. [Prior report](../generalization/REPORT.md) · [KNN distance and voting documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

The search reused the saved three group-disjoint development folds and kept their Kaggle and field validation partitions fixed. The selection score gives equal weight to the mean Kaggle and field CV macro F1. The two validation partitions were used for final development reporting. The previous Kaggle and external tests were kept as historical results. The earlier KNN, Logistic Regression, and K-Means files remain in their original directories.

The staged search compared compact foreground and shape descriptors, joint color alone, global color with HOG, LBP and edges, and combinations with local visual words. After screening the representations, it crossed PCA sizes 64, 128, 256 and no PCA with all six neighborhood sizes, all three distances, and both voting schemes for the two leading representations. It also tried a small set of feature weight changes and a neighbor pool with a larger proportion of field photos.

Two supervised preprocessing experiments tested ANOVA feature selection and a regularized LDA projection. LDA learns directions that put training photos from the same label closer together. The polynomial variant uses Nyström features before LDA to capture nonlinear relationships between handcrafted descriptors. The output classifier remains KNN. A balanced variant gives field rows more influence in fitting the feature space while keeping each photo once in the final neighbor pool. Every learned transformation, including the local codebook, was fitted inside the relevant training fold. [Candidate CSV](experiments.csv) · [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) · [LDA projection](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

The fresh field sample uses iNaturalist observations from 2020–2021, with photographers blocked if they appeared in the supplemental training set or either previous field test. The audit compared candidates with all raw Kaggle photos and the three existing field collections using exact, pixel, and perceptual matching. It excluded one candidate and retained 191 photos across all five classes. The manifest records the photo hash, originating observation, label, and attribution. [Fresh manifest](fresh_manifest.csv) · [Overlap exclusion](overlap_exclusions.csv)

## Selection and development results

| Search result | Balanced CV macro F1 |
|---|---:|
| Archived KNN refitted on the same folds | 53.17% |
| Best ordinary KNN with 64 PCA components | 57.09% |
| Best ordinary KNN with 128 PCA components | 57.35% |
| Best ordinary KNN with 256 PCA components | 57.77% |
| Best ordinary KNN without PCA | 57.33% |
| ANOVA selection of 384 compact features | 58.99% |
| Best plain LDA projection after PCA | 62.79% |
| Polynomial Nyström and LDA | 65.97% |
| **Polynomial Nyström and LDA, with balanced feature fitting** | **67.00%** |

The selected model uses the existing center crop, standardized joint color, HOG, LBP, multiscale edges, and local visual words. The five weights are `(2.0, 0.55, 0.7, 0.7, 1.25)`. It uses 500 landmarks in the degree-2 Nyström map with gamma 0.18, then projects to four label-aware LDA coordinates without PCA. It votes across 15 Manhattan neighbors weighted by inverse distance. It retains all 3,467 original training photos in the neighbor pool. The balanced transformation fit repeats field rows within each training fold. Giving more weight to color or shape performed below the ordinary 256-component compact baseline, as did reducing the Kaggle portion of the neighbor pool. [Full candidate search](experiments.csv) · [Selected parameters](selected.csv)

Each cell is **accuracy / macro F1**, in percent. The CV numbers are means of the three fold scores computed with the actual scikit-learn classifiers.

| Source and split | Archived KNN | Selected KNN |
|---|---:|---:|
| Kaggle train | 71.73 / 71.59 | 100.00 / 100.00 |
| Field train | 54.00 / 53.20 | 100.00 / 100.00 |
| Kaggle CV | 64.44 / 64.24 | 72.19 / 72.05 |
| Field CV | 44.00 / 42.10 | 63.33 / 61.96 |
| Kaggle validation | 64.50 / 64.28 | 71.32 / 71.30 |
| Field validation | 49.11 / 47.62 | 62.50 / 62.04 |

The selected model's training score includes each photo as its own exact match. Excluding the query's group from the vote reduces training accuracy to 77.69% on Kaggle and 89.56% on field photos. Those are diagnostic figures because the transform has still seen the training rows. Grouped CV gives the held-out estimate. [Verified CV](verified_cv.csv) · [Development metrics](development_metrics.csv) · [Development per-class metrics](development_per_class.csv)

On held-out field CV rows, 55.57% of the original KNN's neighbors came from the field source, although that source made up about 13% of the training pool. Only 32.59% of neighbors had the query's flower label. For the selected geometry, those shares were 15.07% and 54.89%. This is consistent with the old geometry grouping photos by source appearance even when the flower labels differ. [Neighbor audit](neighbor_audit.csv)

## Fresh independent test

The model was selected and saved before this paired evaluation. The existing Logistic Regression was included as a fixed comparison on exactly the same 191 photos.

| Model | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Archived KNN | 82 | 42.93% | 40.05% |
| **Selected KNN** | **121** | **63.35%** | **62.93%** |
| Frozen Logistic Regression | 119 | 62.30% | 62.05% |

The selected KNN recognized 39 more photos than the archived KNN and two more than Logistic Regression on this sample. It fell short of 70%. The narrow margin over Logistic Regression describes this sample. The earlier 52.00% KNN and 74.67% Logistic Regression scores came from a different 150-image cohort. [Fresh summary](fresh_metrics.csv) · [Per-photo predictions](fresh_predictions.csv)

| Flower | Support | Archived recall | Selected precision | Selected recall | Selected F1 |
|---|---:|---:|---:|---:|---:|
| Daisy | 40 | 62.50% | 69.77% | 75.00% | 72.29% |
| Dandelion | 40 | 70.00% | 51.72% | 75.00% | 61.22% |
| Rose | 39 | 35.90% | 73.68% | 71.79% | 72.73% |
| Sunflower | 33 | 39.39% | 78.26% | 54.55% | 64.29% |
| Tulip | 39 | 5.13% | 51.72% | 38.46% | 44.12% |

The saved model runs in the existing Streamlit generalization mode. Desktop and mobile Chromium checks exercised the model switch, JPEG upload, five predicted probabilities, and invalid-image recovery. The earlier models remain in the [generalization model directory](../generalization/models/). [New KNN bundle](knn_classifier.joblib) · [Fresh confusion counts](fresh_confusions.csv) · [Run commands](../../../README.md)
