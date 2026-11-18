# K-Means follow-up

The archived model used full-frame compact color and shape descriptors, a `StandardScaler` per group, group weights divided by the square root of group width, and 32 PCA components. It fit five clusters with 15 initializations. A one-to-one assignment maximized the number of matching training labels. The study reproduced the archived validation predictions and kept the original model files. [Archived configuration](../generalization/selected.csv)

The study reused the grouped development folds for 3,017 Kaggle training photos and 450 supplemental field training photos. The saved validation rows contain 645 Kaggle and 112 field photos. The scaler, patch vocabulary, PCA, and centroids were fitted without flower labels on each fitting fold. For five clusters, the mapping kept the one-to-one assignment. For 10 and 15 clusters, each cluster took the most frequent fitting label, with ties resolved in the fixed class order. An eligible final mapping needed to emit all five flower names. [Saved folds](../generalization/development_split.csv)

The feature screen compared joint color, HOG, LBP, multiscale edges, local visual words, compact foreground descriptors, and combinations. For the three leading representations it crossed PCA sizes 20, 40, 80, 120, and no PCA with 5, 10, and 15 clusters. The follow-up alternatives compared group weights, group L2 normalization versus standardized groups, and 10 versus 20 k-means++ starts with seed 42. A source-weighted variant used the photo source to balance the centroid loss. Flower labels were used only for the post-fit cluster names and the development diagnostics. [K-Means parameters](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

Selection used the mean of the two sources' grouped CV macro F1 scores. The GUI promotion rule required an improvement in both that score and the equal-source validation macro F1. Validation metrics were recorded for every candidate. Only the selected candidate's validation score entered the promotion check. The existing test sets were excluded from this study.

Silhouette uses Euclidean distance in the fitted feature space. Partitions above 800 rows use a fixed-seed sample of 800. ARI and NMI compare raw cluster IDs with the flower labels inside each source and fold. Accuracy, macro F1, and recall use the training-mapped labels. Higher silhouette can indicate compact groups that follow attributes such as color or background. ARI, NMI, and the mapped scores show how closely those groups align with flower classes. [Clustering evaluation](https://scikit-learn.org/stable/modules/clustering.html)

## Feature and parameter results

The best configuration among 73 candidates used **local visual words, 20 PCA components, 15 clusters, and 10 k-means++ starts**. It used the existing center crop and the internally L2-normalized patch residual representation. The codebook learns 24 patch centers from small HOG and color descriptors and sums signed differences from those centers for each image. The final centroids used all training photos with equal weights.

| PCA dimensions | Best cluster count for local visual words | Balanced CV macro F1 |
|---:|---:|---:|
| **20** | **15** | **37.76%** |
| 40 | 10 | 36.55% |
| 80 | 15 | 36.11% |
| 120 | 10 | 36.94% |
| No PCA | 10 | 37.50% |

The best eligible five-cluster model scored 32.66% and the best 10-cluster model scored 37.50%. The compact representation with group L2 normalization peaked at 33.61%. Switching its leading configuration to standardization lowered the score to 30.07% and left four mapped labels. Adding a small amount of joint color to the local descriptor scored 35.69%. Adding LBP and edges scored 37.17%. At the winning geometry, 20 starts scored 36.47% and source-weighted centroids scored 35.96%. [All 73 configurations](experiments.csv) · [Selected configuration](selected.csv)

## Development comparison

Each cell shows **accuracy / macro F1**, in percent.

| Partition | Archived | Selected |
|---|---:|---:|
| Kaggle train | 39.64 / 38.10 | 51.28 / 51.49 |
| Kaggle CV | 37.22 / 35.44 | **49.49 / 49.25** |
| Kaggle validation | 37.05 / 35.57 | **51.78 / 51.97** |
| Field train | 25.11 / 19.35 | 27.33 / 22.19 |
| Field CV | 24.44 / 20.18 | **28.22 / 26.28** |
| Field validation | 22.32 / 18.08 | **32.14 / 28.91** |

The equal-source CV macro F1 rose from **27.81% to 37.76%**, and validation macro F1 rose from **26.82% to 40.44%**. Field validation recall improved for dandelion (0.00% to 75.00%), rose (12.50% to 25.00%), and tulip (0.00% to 8.33%). Daisy fell from 66.67% to 29.17% and sunflower from 37.50% to 18.75%. All five labels remain available in the saved mapping. [All five per-class recalls and F1](development_per_class.csv) · [Grouped CV confusion matrices](cv_confusions.png) · [Validation confusion matrices](validation_confusions.png) · [Counts and row percentages](development_confusions.csv)

| Partition | Silhouette, old → selected | ARI, old → selected | NMI, old → selected |
|---|---:|---:|---:|
| Kaggle CV | 0.059 → 0.083 | 0.135 → 0.098 | 0.173 → 0.178 |
| Kaggle validation | 0.058 → 0.085 | 0.128 → 0.096 | 0.178 → 0.203 |
| Field CV | 0.062 → 0.048 | 0.014 → 0.038 | 0.099 → 0.178 |
| Field validation | 0.074 → 0.001 | 0.023 → 0.049 | 0.127 → 0.204 |

The lower Kaggle ARI and the almost-zero field validation silhouette show that the mapped accuracy gain comes with weak separation in the fitted space. K-Means minimizes distances to centroids. It can divide one species into several clusters or group distinct species that look similar. The post-fit majority mapping can combine several centers into one predicted class, so mapped F1 and raw clustering diagnostics can move in different directions. [Per-fold clustering metrics](fold_metrics.csv)

Streamlit generalization mode uses the promoted snapshot because both development macro F1 criteria passed. A Chromium upload test confirmed that the app matched the saved pipeline's daisy prediction and displayed cluster 12 on the project example. The earlier bundle is still available with the archive flag. The previously published 27.33% remains a historical test result. [Saved K-Means pipeline](kmeans.joblib) · [Cluster-to-class mapping](cluster_labels.json) · [Training votes per cluster](cluster_train_counts.csv) · [Run commands](../../../README.md)
