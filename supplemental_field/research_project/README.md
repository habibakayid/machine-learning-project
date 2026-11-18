# Five-class flower classification

Part B classifies daisy, dandelion, rose, sunflower, and tulip using handcrafted features. The default model suite fits Kaggle images only. Supplemental field-training experiments are separately identified.

## Launch the final demo

Open the folder that contains gui, models and part_b_images. Run this command in PowerShell.

    .\.venv\Scripts\python.exe -m streamlit run gui/image_app.py

Select a model, upload a JPEG, PNG or WebP, or try the dataset example. The supervised models display five probability estimates. K-Means displays its cluster and training-mapped label. The Training data selector switches between primary Kaggle-only models and the supplemental Kaggle + field experiment.

| Default primary model | Bundle |
|---|---|
| Nyström Logistic Regression | [selected.joblib](part_b_images/rebuild/final_optimization/primary/logistic_regression/selected.joblib) |
| Nyström–LDA KNN | [selected.joblib](part_b_images/rebuild/final_optimization/primary/knn_classifier/selected.joblib) |
| K-Means | [selected.joblib](part_b_images/rebuild/final_optimization/primary/k-means/selected.joblib) |

The supplemental suite is available in the same selector. The CLI flag --generalization starts with that suite selected. Archive flags after the Streamlit argument separator remain available as --archived-rebuild, --archived-knn, and --archived-kmeans. Restore a missing bundle from the project backup using the path in the model manifest.

## Environment

The existing .venv uses Python 3.11. For a fresh environment, run py -3.11 -m venv .venv, followed by the dependency command below.

    .\.venv\Scripts\python.exe -m pip install -r part_b_images/rebuild/requirements.txt

The raw images are stored in part_b_images/data/raw/flowers/. Field manifests contain observation URLs, attribution, photo licenses and file hashes. [Kaggle Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) · [Supplemental training manifest](part_b_images/rebuild/generalization/field_manifest.csv)

## Final optimization evidence

- [Final report](part_b_images/rebuild/final_optimization/FINAL_REPORT.md)
- [Baseline before experiments](part_b_images/rebuild/final_optimization/BASELINE_REPORT.md)
- [Model selection](part_b_images/rebuild/final_optimization/FINAL_MODEL_SELECTION.md)
- [Training, CV and validation metrics](part_b_images/rebuild/final_optimization/FINAL_MODEL_METRICS.csv)
- [All experiments](part_b_images/rebuild/final_optimization/ALL_EXPERIMENTS.csv)
- [Per-class metrics](part_b_images/rebuild/final_optimization/FINAL_PER_CLASS_METRICS.csv)
- [Data integrity](part_b_images/rebuild/final_optimization/DATA_INTEGRITY_AUDIT.md)
- [Overfitting and KNN diagnostics](part_b_images/rebuild/final_optimization/OVERFITTING_ANALYSIS.md)
- [GUI verification](part_b_images/rebuild/final_optimization/GUI_VERIFICATION.md)
- [Model hashes and fitting sources](part_b_images/rebuild/final_optimization/MODEL_MANIFEST.json)

The course PDF's restriction on AI-written submitted code still needs resolution with the TA. Primary training now uses one dataset, and the technical demo is verified separately from submission eligibility.

## Reproduce controlled development experiments

Run one family at a time from the root. Choose primary or supplemental and one of Logistic Regression, KNN Classifier, or K-Means.

    .\.venv\Scripts\python.exe part_b_images/rebuild/final_optimization/optimize.py --scope primary --family "Logistic Regression"

After the six studies have results, apply the documented development-only stability correction and export the consolidated evidence.

    .\.venv\Scripts\python.exe part_b_images/rebuild/final_optimization/reselect.py
    .\.venv\Scripts\python.exe part_b_images/rebuild/final_optimization/summarize.py
    .\.venv\Scripts\python.exe part_b_images/rebuild/final_optimization/write_reports.py

The overlap audit can be repeated by importing final_review/audit_data.py, setting its OUT directory to final_optimization, and calling audit_data(). Historical tests were already accessed. The current optimization makes no new historical test predictions.

## Preserved experiment history

[Original rebuild](part_b_images/rebuild/REPORT.md) · [Generalization study](part_b_images/rebuild/generalization/REPORT.md) · [Regularization study](part_b_images/rebuild/logistic_study/REPORT.md) · [KNN study](part_b_images/rebuild/knn_study/REPORT.md) · [K-Means study](part_b_images/rebuild/kmeans_study/REPORT.md) · [Earlier final audit](part_b_images/rebuild/final_review/FINAL_AUDIT.md) · [Historical test cohorts](part_b_images/rebuild/final_review/TEST_COHORTS.md)
