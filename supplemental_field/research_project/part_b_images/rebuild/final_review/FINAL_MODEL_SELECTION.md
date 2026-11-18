# Final Part B model selection

Selection uses the existing grouped development results and the study-specific promotion rules. Test scores describe already frozen models and did not enter these decisions.

| GUI choice | Final bundle, relative to the project root | Model input → estimator features |
|---|---|---:|
| Logistic Regression | [generalization Nyström](../generalization/models/logistic_regression.joblib) | 10,299 → 1,300 |
| KNN Classifier | [KNN study winner](../knn_study/knn_classifier.joblib) | 10,299 → 4 |
| K-Means | [Promoted K-Means snapshot](../kmeans_study/promoted_kmeans.joblib) | 5,488 → 20 |

The [model manifest](MODEL_MANIFEST.json) records exact paths, SHA-256 hashes, feature groups, dimensions, steps, and estimator parameters. All three use the same EXIF-corrected, 128 × 128 center-crop procedure and scaled pixels. The first two combine joint color, HOG, LBP, multiscale edge descriptors, and local visual words. The 10,299 input values include raw patch descriptors. Pooling the patch descriptors reduces the full representation to 6,131 dimensions before the learned similarity transform. K-Means uses the local descriptor alone. It pools 5,488 patch input values into 1,320 residual features before PCA.

## Logistic Regression

| Candidate | Equal-source CV macro F1 | Kaggle train–validation gap | Field train–validation gap | Decision |
|---|---:|---:|---:|---|
| Original rebuilt Nyström, Kaggle training | 60.20% | 22.87 pp | Unavailable | Archived benchmark |
| Field-trained Nyström, C=18 | 68.52% | 22.95 pp | 42.87 pp | **Deployed** |
| Regularized Nyström, C=3 and source/class weights | 70.25% | 14.12 pp | 44.65 pp | Preserved candidate |
| Compact LR, 738 features | 65.95% | 8.13 pp | 29.95 pp | Preserved benchmark |

The field-trained model uses a degree-2 polynomial Nyström map, gamma 0.18, 1,300 landmarks, and C=18. The C=3 model won its experimental search, and it scored 69.23% versus 64.10% for the deployed model on the same historical 156-image cohort. The recorded promotion rule required higher development CV macro F1 and smaller gaps in **both** sources. The C=3 field gap grew, so its saved bundle remains a candidate. The Compact model illustrates the tradeoff. Its smaller gaps come with lower balanced CV macro F1. [Regularization study](../logistic_study/REPORT.md) · [Generalization study](../generalization/REPORT.md)

## KNN

The chosen classifier uses 500 Nyström landmarks, polynomial degree 2, gamma 0.18, four LDA coordinates, 15 Manhattan neighbors, and distance-weighted voting. Equal-source CV macro F1 rose from 53.17% for the archived compact KNN to 67.00%. The study fits the group scalers, vocabulary, Nyström map, and LDA using only the fitting fold. LDA sees fitting labels only. Fold-local field repetition affects the transformation fit. The final neighbor pool contains the 3,467 original fitting images once each. The archived compact KNN is available through the demo archive flag. [KNN study](../knn_study/REPORT.md) · [Verified fold means](../knn_study/verified_cv.csv)

## K-Means

The chosen pipeline uses an unsupervised local patch vocabulary, 20 PCA components, 15 image clusters, 10 k-means++ starts, and seed 42. Flower labels enter after centroid fitting, through the training-only majority cluster mapping. The audited training vote table contains 3,467 assignments and verifies all 15 majorities and all five named output classes.

The candidate met both promotion checks. Equal-source CV macro F1 improved from 27.81% to 37.76%, and validation macro F1 from 26.82% to 40.44%. Best eligible K=5 CV macro F1 in the follow-up was 32.66%. The original K=5 compact bundle remains selectable as an archive. The course PDF calls for a majority label per cluster and a silhouette score. It specifies 3–5 **classes** and gives no fixed cluster count. [K-Means study](../kmeans_study/REPORT.md) · [Training votes](../kmeans_study/cluster_train_counts.csv) · [Mapping](../kmeans_study/cluster_labels.json)

| Partition | Silhouette | ARI | NMI |
|---|---:|---:|---:|
| Kaggle grouped CV | 0.083 | 0.098 | 0.178 |
| Kaggle validation | 0.085 | 0.096 | 0.203 |
| Field grouped CV | 0.048 | 0.038 | 0.178 |
| Field validation | 0.001 | 0.049 | 0.204 |

The low field silhouette shows weak separation in the actual fitted feature space. The higher mapped F1 measures a different property. Several overlapping clusters can map to the same flower label. [Raw clustering diagnostics](../kmeans_study/fold_metrics.csv)
