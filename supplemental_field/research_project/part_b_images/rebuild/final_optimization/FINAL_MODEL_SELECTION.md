# Final model selection

The initial policy was recorded before fitting. The [selection policy](SELECTION_POLICY.md) documents one development-only correction to the stability rule, with the initial policy and decisions preserved. The primary suite fits Kaggle only and is the application default. Supplemental training uses iNaturalist as a separately identified generalization experiment. Numerical target ranges do not enter selection.

| Suite | Algorithm | Selected candidate | CV selection macro F1 | Promoted over baseline |
|---|---|---|---:|---|
| primary | Logistic Regression | C=3 | 77.35% | Yes |
| primary | KNN Classifier | neighbor sweep/euclidean/k=31/uniform | 76.82% | Yes |
| primary | K-Means | Local PCA20 clusters=10 | 48.87% | Yes |
| supplemental | Logistic Regression | C=5 | 69.67% | Yes |
| supplemental | KNN Classifier | neighbor sweep/euclidean/k=31/distance | 67.22% | Yes |
| supplemental | K-Means | baseline | 37.76% | No |

Primary LR retains degree-2 polynomial Nyström, gamma=0.18 and 1,300 landmarks, with C reduced to 3. Supplemental LR uses C=5 with equal per-image sample weights. Both retain all five feature groups. Primary KNN uses a 500-landmark degree-2 map and a four-dimensional supervised LDA projection fitted inside every CV fold, 31 Euclidean neighbors and uniform voting. Supplemental KNN keeps the same field-balanced fitting representation as its baseline, with 31 Euclidean neighbors and distance voting. No PCA model beat the selected LDA configurations under the promotion checks. Local visual words with PCA20 use 10 primary clusters and 15 supplemental clusters; each cluster is mapped from fitting labels only.

## Primary / Logistic Regression

Bundle [selected.joblib](primary/logistic_regression/selected.joblib). Fitting counts {'kaggle': 3017}. Preparation crop, groups ['color', 'hog', 'lbp', 'edges', 'local'], weights [2.0, 0.55, 0.7, 0.7, 1.25], scaling standard. Steps ['feature_groups', 'similarities', 'estimator']. Input width 10299, estimator width 1300. Estimator settings {'C': 3.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': 0.0, 'max_iter': 1000, 'n_jobs': None, 'penalty': 'deprecated', 'random_state': 42, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}.

## Primary / KNN Classifier

Bundle [selected.joblib](primary/knn_classifier/selected.joblib). Fitting counts {'kaggle': 3017}. Preparation crop, groups ['color', 'hog', 'lbp', 'edges', 'local'], weights [2.0, 0.55, 0.7, 0.7, 1.25], scaling standard. Steps ['feature_groups', 'similarities', 'projection', 'estimator']. Input width 10299, estimator width 4. Estimator settings {'algorithm': 'brute', 'leaf_size': 30, 'metric': 'euclidean', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 31, 'p': 2, 'weights': 'uniform'}.

## Primary / K-Means

Bundle [selected.joblib](primary/k-means/selected.joblib). Fitting counts {'kaggle': 3017}. Preparation crop, groups ['local'], weights [1.0], scaling l2. Steps ['feature_groups', 'pca', 'estimator']. Input width 5488, estimator width 20. Estimator settings {'algorithm': 'lloyd', 'copy_x': True, 'init': 'k-means++', 'max_iter': 300, 'n_clusters': 10, 'n_init': 10, 'random_state': 42, 'tol': 0.0001, 'verbose': 0}.

## Supplemental / Logistic Regression

Bundle [selected.joblib](supplemental/logistic_regression/selected.joblib). Fitting counts {'field': 450, 'kaggle': 3017}. Preparation crop, groups ['color', 'hog', 'lbp', 'edges', 'local'], weights [2.0, 0.55, 0.7, 0.7, 1.25], scaling standard. Steps ['feature_groups', 'similarities', 'estimator']. Input width 10299, estimator width 1300. Estimator settings {'C': 5.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': 0.0, 'max_iter': 1000, 'n_jobs': None, 'penalty': 'deprecated', 'random_state': 42, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}.

## Supplemental / KNN Classifier

Bundle [selected.joblib](supplemental/knn_classifier/selected.joblib). Fitting counts {'field': 450, 'kaggle': 3017}. Preparation crop, groups ['color', 'hog', 'lbp', 'edges', 'local'], weights [2.0, 0.55, 0.7, 0.7, 1.25], scaling standard. Steps ['feature_groups', 'similarities', 'projection', 'estimator']. Input width 10299, estimator width 4. Estimator settings {'algorithm': 'brute', 'leaf_size': 30, 'metric': 'euclidean', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 31, 'p': 2, 'weights': 'distance'}.

## Supplemental / K-Means

Bundle [selected.joblib](supplemental/k-means/selected.joblib). Fitting counts {'field': 450, 'kaggle': 3017}. Preparation crop, groups ['local'], weights [1.0], scaling l2. Steps ['feature_groups', 'pca', 'estimator']. Input width 5488, estimator width 20. Estimator settings {'algorithm': 'lloyd', 'copy_x': True, 'init': 'k-means++', 'max_iter': 300, 'n_clusters': 15, 'n_init': 10, 'random_state': 42, 'tol': 0.0001, 'verbose': 0}.

Earlier bundles remain in their original folders. Unconstrained CV leaders are preserved as best_cv_candidate.joblib inside each study folder. The selected.joblib implements all promotion checks. The [experiment CSV](ALL_EXPERIMENTS.csv) records rejection reasons. The [model manifest](MODEL_MANIFEST.json) pins hashes and fitting sources.
