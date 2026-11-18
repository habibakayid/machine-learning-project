# Final project structure

| Location | Purpose |
|---|---|
| [README](../../../README.md) | Final launch command, model paths and audit entry points |
| [gui/image_app.py](../../../gui/image_app.py) | Streamlit model selection, image validation and preview, shared inference, and output display |
| [dataset_preparation.py](../dataset_preparation.py) | Image audit, distinct-photo grouping, frozen Kaggle splits |
| [feature_extraction.py](../feature_extraction.py) and [local_features.py](../local_features.py) | Shared resizing, handcrafted global descriptors, and patch vocabulary |
| [generalization_features.py](../generalization_features.py) | Compact foreground and shape descriptors for benchmark bundles |
| [train_models.py](../train_models.py) | Shared scikit-learn pipelines and saved-model inference |
| [generalization](../generalization/) | Field development splits, mixed-source Logistic Regression and Compact benchmark, plus their saved reports |
| [logistic_study](../logistic_study/) | Regularized C=3 candidate, selection evidence, and separate 156-photo test |
| [knn_study](../knn_study/) | Selected Nyström–LDA KNN, grouped development metrics, and 191-photo comparison |
| [kmeans_study](../kmeans_study/) | Selected and promoted K-Means bundles, training mapping votes, and development clustering metrics |
| [rebuild/results](../results/) and [rebuild/models](../models/) | Original rebuild predictions, metrics, split manifests, and earlier bundles |
| [Original models](../../../models/) and [original Part B scripts](../../scripts/) | Pre-rebuild experiments and their historical artifacts |
| [final_review](./) | Final selection, verified tables, confusion matrices, and demo fixtures |

`audit_data.py` verifies file hashes and split boundaries. `audit_results.py` checks existing confusion counts, consolidates stored predictions, regenerates the final figures, and checks inference with the saved development bundles. The scripts use the existing environment. Neither script fits a model or predicts a historical test cohort.

Keep the earlier study records together with their manifests. A historical result is traceable by model ID, cohort, predictions or original matrix, and provenance. The model hashes in [MODEL_MANIFEST.json](MODEL_MANIFEST.json) pin the files used in this audit.
