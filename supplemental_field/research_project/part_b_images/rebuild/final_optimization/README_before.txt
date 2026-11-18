# Five-class flower classification

Part B classifies daisy, dandelion, rose, sunflower, and tulip from handcrafted image features. The local app loads saved scikit-learn pipelines.

## Run the final demo

Open the folder that contains `gui`, `models`, and `part_b_images`. Run this command in PowerShell from that folder.

```powershell
.\.venv\Scripts\python.exe -m streamlit run gui/image_app.py
```

Select a model and upload a JPEG, PNG, or WebP, or choose **Try a dataset example**. Logistic Regression and KNN show five estimated class probabilities. K-Means displays its cluster ID and the flower label based on training votes.

| Default model | Saved bundle |
|---|---|
| Field-trained Nyström Logistic Regression | [logistic_regression.joblib](part_b_images/rebuild/generalization/models/logistic_regression.joblib) |
| Nyström–LDA Manhattan KNN | [knn_classifier.joblib](part_b_images/rebuild/knn_study/knn_classifier.joblib) |
| Local visual words K=15 K-Means | [promoted_kmeans.joblib](part_b_images/rebuild/kmeans_study/promoted_kmeans.joblib) |

Restore a missing bundle from the audited project copy. Use `-- --archived-rebuild` to select the earlier rebuilt suite, `-- --archived-knn` for the compact field KNN, or `-- --archived-kmeans` for the compact K=5 model.

## Environment and datasets

The project uses Python 3.11 and its existing `.venv`. If the environment needs to be recreated, run `py -3.11 -m venv .venv` and install [the pinned dependencies](part_b_images/rebuild/requirements.txt) with `.\.venv\Scripts\python.exe -m pip install -r part_b_images/rebuild/requirements.txt`.

The raw Kaggle images are in `part_b_images/data/raw/flowers/`. The field manifests record the iNaturalist observation URLs, licenses, filenames and hashes. [Kaggle Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) · [Field-training manifest](part_b_images/rebuild/generalization/field_manifest.csv) · [All partition counts](part_b_images/rebuild/final_review/DATA_SPLITS.csv)

## Final audit and evidence

- [Final audit](part_b_images/rebuild/final_review/FINAL_AUDIT.md)
- [Model selection](part_b_images/rebuild/final_review/FINAL_MODEL_SELECTION.md)
- [Separate development tables](part_b_images/rebuild/final_review/DEVELOPMENT_TABLES.md)
- [Historical test cohorts](part_b_images/rebuild/final_review/TEST_COHORTS.md)
- [Confusion matrices](part_b_images/rebuild/final_review/CONFUSION_MATRIX_AUDIT.md)
- [Demo checklist](part_b_images/rebuild/final_review/FINAL_DEMO_CHECKLIST.md)
- [Project structure](part_b_images/rebuild/final_review/FINAL_PROJECT_STRUCTURE.md)

To regenerate the audit CSVs, hashes and figures, run `.\.venv\Scripts\python.exe part_b_images/rebuild/final_review/audit_data.py` and `.\.venv\Scripts\python.exe part_b_images/rebuild/final_review/audit_results.py` from the root. These commands audit fixed files and run development inference. Historical test figures come from saved predictions or archived integer matrices.

Earlier experiments remain in the [original rebuild](part_b_images/rebuild/REPORT.md), [generalization](part_b_images/rebuild/generalization/REPORT.md), [Logistic Regression](part_b_images/rebuild/logistic_study/REPORT.md), [KNN](part_b_images/rebuild/knn_study/REPORT.md), and [K-Means](part_b_images/rebuild/kmeans_study/REPORT.md) reports.
