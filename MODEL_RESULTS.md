# Frozen primary model results

All percentages below are Kaggle-only. Every cell is Accuracy / Macro F1. These are existing development estimates, not new external test results.

| Model | Train | Grouped CV | Validation |
|---|---:|---:|---:|
| K-Means | 50.98% / 49.82% | 49.29% / 48.87% | 53.49% / 53.22% |
| KNN Classifier | 85.68% / 85.58% | 77.00% / 76.82% | 77.52% / 77.09% |
| Logistic Regression | 94.03% / 94.01% | 77.49% / 77.35% | 76.90% / 76.55% |

CV is the mean of three separately fitted grouped folds; class reports and confusion matrices pool one held-out prediction per fitting image. Counts can therefore differ very slightly from a mean-fold percentage. Class order is Daisy, Dandelion, Rose, Sunflower, Tulip; confusion rows are true labels, columns predicted labels.

LR: C=3, polynomial degree 2, gamma=0.18, 1,300 Nyström landmarks, all five handcrafted groups. KNN: 500 polynomial landmarks, LDA4, 31 Euclidean neighbors, uniform vote. K-Means: local visual words, PCA20, 10 clusters, n_init=10, random_state=42, training-majority mapping. All exact parameters and SHA256 values are in app/MODEL_MANIFEST.json.

Remaining ordinary train–validation gaps: LR 17.13 pp, KNN 8.16 pp, K-Means -2.51 pp. A low K-Means gap reflects poor alignment of clusters to taxonomy, not a strong classifier. Validation silhouette=0.07865, ARI=0.11687, NMI=0.19603. The 100% resubstitution score from earlier distance-voting supplemental KNN is not the primary model score.

Weak validation classes: primary LR and KNN have their lowest F1 on Daisy; primary K-Means on Rose (37.45% F1). Common errors include Daisy → Dandelion and Tulip → Rose. Against the respective primary baselines, KNN Tulip → Rose errors fell by 17 and K-Means Dandelion → Daisy errors by 46. The complete before/after counts are in primary/results/ERROR_CHANGES.csv.

All five per-class precision, recall and F1 values and raw/row-normalized confusion counts are in primary/results/. Primary figures contain counts and row percentages for train, grouped CV and validation. Historical learning curves are explicitly prefixed historical; they correspond to an earlier LR recipe, not C=3.

Supplemental results are historical and separate. Field Validation Accuracy for the selected mixed-source variants was LR 55.36%, KNN 63.39%, K-Means 32.14%. They were fitted with 450 field photos in addition to Kaggle, with 112 field validation photos. These are not primary course results. Their exact per-source training/CV/validation, macro F1, observer folds, models and earlier independent cohorts remain under supplemental_field/research_project.

No test set was rerun for model selection during packaging. The current release includes no newly collected common untouched external test.
