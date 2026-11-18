# Reproduction and training guide

All commands below assume the release root as the PowerShell working directory and Python 3.11 in `.venv`. Install and import the Kaggle dataset as described in README.md. Training was not executed while packaging.

## Frozen primary recipes

`tools/train_primary.py` is a new handover wrapper around the actual original extractor and exact saved estimator/pipeline configurations. It clones rather than reuses fitted transformations. `tools/common.py` loads `primary/development_split.csv`, attaches hashes from the cleaned manifest, and checks Kaggle-only membership, group exclusion and the original train/validation counts. Fold assignments are the existing ones; no repartitioning occurs.

```powershell
.\.venv\Scripts\python.exe tools\train_primary.py --output runs\primary_refit_01 --check-only
.\.venv\Scripts\python.exe tools\train_primary.py --output runs\primary_refit_01
```

Only the second command fits. It writes three new joblib files, per-model CV predictions and training_results.json with fold-level and pooled metrics, per-class reports and confusion counts. It does not overwrite `app/models`. LR uses C=3; KNN uses 31 Euclidean neighbors/uniform voting with fold-local LDA; K-Means uses 10 clusters. Saved parameters are read from the unchanged templates. There is no new hyperparameter selection here.

## Actual original experiment implementations

`supplemental_field/research_project` preserves all original relative paths. Original files include:

- `part_b_images/rebuild/dataset_preparation.py`: image validity, hashes, duplicate/related grouping, cleaned partitions.
- `feature_extraction.py`, `local_features.py`, `generalization_features.py`: actual image and local/foreground features.
- `train_models.py`: original baseline feature/model comparisons. Its defaults are historical, not the final selected configurations.
- `generalization_study.py`: original mixed/source-balanced experiments, original field grouping and compact benchmark.
- `logistic_study.py`, `knn_study.py`, `kmeans_study.py`: actual previous model studies.
- `final_optimization/optimize.py`: controlled primary/supplemental final stages.
- `final_optimization/reselect.py`: documented stability amendment from stored development predictions.
- `final_optimization/summarize.py`, `write_reports.py`: original consolidated exports.
- `evaluate_models.py`: historical evaluation; `--learning-curve` explicitly refits and must not be confused with frozen inference.
- `part_b_images/scripts/01_...20_...py`: preserved earliest pipeline iterations. They use their working directory and some earlier array assumptions. They are not the recommended final-model training route.

Do not run those mutation-capable scripts inside the distributed archive. Create a scratch copy:

```powershell
.\.venv\Scripts\python.exe tools\import_dataset.py --archive 'D:\Downloads\flowers-recognition.zip' --research
.\.venv\Scripts\python.exe tools\research_workspace.py --output runs\research_replay_01
```

The next loop executes the actual original controlled experiments in that new workspace, retaining the fixed development folds. It is a costly optional rerun, NOT a packaging or app-launch command:

```powershell
foreach ($scope in @('primary','supplemental')) {
    foreach ($family in @('Logistic Regression','KNN Classifier','K-Means')) {
        .\.venv\Scripts\python.exe runs\research_replay_01\part_b_images\rebuild\final_optimization\optimize.py --scope $scope --family $family
        if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $scope / $family" }
    }
}
.\.venv\Scripts\python.exe runs\research_replay_01\part_b_images\rebuild\final_optimization\reselect.py
.\.venv\Scripts\python.exe runs\research_replay_01\part_b_images\rebuild\final_optimization\summarize.py
.\.venv\Scripts\python.exe runs\research_replay_01\part_b_images\rebuild\final_optimization\write_reports.py
```

The original summarizer's image checks will use the restored Kaggle data. Existing research reports can contain historical local links or references to deliberately omitted photo previews. They are preserved as source evidence, not silently rewritten. Rerunning updates the scratch results only. `final_optimization/verify_artifacts.py` historically checks all documentation links and GUI records too; omitted legacy previews mean it is not the distribution verifier. Use `tools/verify_package.py` for this package.

For earlier individual studies, their actual entry points and archived reports describe their stages. For example run `generalization_study.py --help`, `logistic_study.py --help`, or `knn_study.py --help` in the scratch copy. `kmeans_study.py` is executable directly and performs its older, larger search; it is not required to reproduce the three selected recipes.

## Evaluation without training

```powershell
.\.venv\Scripts\python.exe tools\evaluate_primary.py --partition both --output runs\saved_model_eval_01
```

This evaluates only fitting/validation rows using the frozen bundles. It does not estimate new CV scores or call fit. Exact expected development tables are already included. Historical test results remain archived and must not drive new choices. If rerunning a true new test later, freeze selection first and record that cohort's provenance independently.

## Optional repeated data audit

```powershell
.\.venv\Scripts\python.exe tools\audit_dataset.py --output runs\image_audit_01
```

This runs the actual cleaning/grouping implementation into a new output directory. Inspect and compare outputs before replacing any split. `primary/original_splits` preserves the old partitions and cleaning decisions; `primary/*_manifest.csv` and `primary/development_split.csv` describe the final experiment. Historical 64/96-pixel array outputs are deliberately not distributed; the current archived preprocessing entry points and feature arrays are retained, but exact reconstruction of every early iteration has not been rerun or guaranteed.

## Reproducibility bounds

Seeds, folds, model settings, feature implementations, Python dependency versions and saved predictions are retained. BLAS, platform or library behavior can produce small numerical changes after refitting; bitwise equality of newly trained bundles is not claimed. The release bundles themselves must always match their recorded SHA256. The new training wrapper has been checked for imports, inputs and cloning without executing training during packaging. Only saved-model inference/evaluation and non-training verification were executed.
