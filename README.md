# Five-Class Flower Classification — Part B

A classical machine learning project for **Daisy, Dandelion, Rose, Sunflower and Tulip**, using handcrafted image features and three algorithms: Logistic Regression, KNN and K-Means. No pretrained networks, transfer learning or deep learning are used.

## Verified primary results

All results below use Kaggle only. Each cell is **Accuracy / Macro F1 (%)**; CV averages three group-separated folds.

| Model | Train | Grouped CV | Validation |
|---|---:|---:|---:|
| Logistic Regression | 94.03 / 94.01 | 77.49 / 77.35 | 76.90 / 76.55 |
| KNN | 85.68 / 85.58 | 77.00 / 76.82 | 77.52 / 77.09 |
| K-Means | 50.98 / 49.82 | 49.29 / 48.87 | 53.49 / 53.22 |

These are development estimates, not a new independent test. See [full results](MODEL_RESULTS.md), [structure](FILE_STRUCTURE.md), [training guide](TRAINING_GUIDE.md), and [limitations](MISSING_ARTIFACTS.md).

## Clone and retrieve model artifacts

Install Git for Windows with Git LFS, then run in PowerShell:

```powershell
git lfs install
git clone https://github.com/malakmohamed172/machine-learning-project.git
cd machine-learning-project
git lfs pull
.\setup_windows.bat
.\run_gui.bat
```

Python **3.11 x64** and the Windows Python Launcher are required. `app/models/logistic_regression.joblib`, `app/models/knn_classifier.joblib`, and `app/models/kmeans.joblib` are the frozen application bundles. The oversized historical `supplemental_field/research_project/models/part_b_knn_96.joblib` uses Git LFS. Downloading GitHub's automatic source ZIP does not guarantee retrieval of LFS content; use a full clone plus `git lfs pull`, or the complete team ZIP from [GitHub Releases](https://github.com/malakmohamed172/machine-learning-project/releases) when available.

The Git working tree includes the licensed supplemental photos omitted by inherited research ignore rules. The giant team ZIP is a release asset, never an ordinary Git commit. Verification inventories describe the frozen team ZIP; Git-specific README, ignore rules and attributes are documented changes and do not change models or experiment evidence.

## Reproducible project guide

Release date: 2026-11-18. Five classes: Daisy, Dandelion, Rose, Sunflower, Tulip. The saved Kaggle-only models are frozen. Packaging did not fit a model, optimize a parameter, or replace any original file. This release contains a working inference application, the actual feature and research code, all 124 final optimization records, historical studies and model bundles, frozen splits, and reproducibility entry points.

**Dataset requirement:** the 4,317 original Kaggle images exist locally in the development project, but Kaggle currently lists the collection's license as Unknown. Original Kaggle photos, resized pixel arrays and photo previews are not redistributed. Download your own copy and run the hash-verifying importer below. See [dataset licensing](datasets/LICENSE_STATUS.md) and [declared omissions](MISSING_ARTIFACTS.md). Supplemental licensed photos retain individual attribution. No Kaggle credentials are included.

## Architecture

- `app/`: the default Streamlit application, inference modules, three exact selected bundles, and 15 functional test expectations. Only Kaggle-only models are selectable.
- `primary/`: frozen Kaggle splits, original and rebuilt manifests, image audit, original feature/training implementations, final primary-only tables and figures. This working tree contains no field fitting data.
- `tools/`: packaging entry points for importing data, auditing, selected-recipe training, saved-model evaluation and verification. Training commands write to new output directories.
- `supplemental_field/research_project/`: an immutable historical research snapshot. It retains the original relative layout, primary baselines needed by the comparisons, all field studies, original training/search/evaluation scripts, 38 historical model files, all 124 optimization candidates and mixed-source results. Its mixed-source reports are explicitly historical research, not primary course results. Copy it to a scratch workspace before running its scripts.
- `supplemental_field/PHOTO_ATTRIBUTION.csv`: per-photo source URLs, observers, license codes, attribution and image hashes for 1,293 licensed field photos. Observer groups and historical cohort boundaries remain in the research manifests.
- `verification/`: source-copy checksums, omissions inventory and executed package checks. [Structure](FILE_STRUCTURE.md), [training guide](TRAINING_GUIDE.md), [metrics](MODEL_RESULTS.md), [handover](TEAM_HANDOVER.md).

## A. Environment setup (Windows)

Use 64-bit Python 3.11 and the Windows Python Launcher. Clone the repository with LFS, or extract the complete release ZIP into a writable folder. Open PowerShell in that project root:

```powershell
.\setup_windows.bat
```

Equivalent commands:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe tools\verify_package.py
```

The full lock file records the previously verified Python environment, including historical plotting dependencies. `requirements.txt` lists direct runtime dependencies. Installing needs internet. No activation or PowerShell policy change is needed. A virtual environment is intentionally not shipped.

## B. Dataset preparation

Download Flowers Recognition, publisher `alxmamaev`, from https://www.kaggle.com/datasets/alxmamaev/flowers-recognition (the recorded inventory corresponds to the 4,317-image collection shown in Version 2). Sign in or accept any current Kaggle terms yourself. Obtain the ZIP; do not substitute another flower dataset.

```powershell
.\.venv\Scripts\python.exe tools\import_dataset.py --archive 'D:\Downloads\flowers-recognition.zip'
```

Alternatively, point to a locally available folder containing the five class directories:

```powershell
.\.venv\Scripts\python.exe tools\import_dataset.py --from-folder 'D:\Datasets\flowers'
```

These paths are examples chosen by the recipient. The importer searches by class/filename, verifies all 4,317 recorded SHA256 values and image readability, restores `primary/part_b_images/data/raw/flowers/`, and installs the GUI's eight dataset examples from those same files. It refuses changed or missing images. The frozen cleaned inventory has 4,307 images after ten exclusions. Do not regenerate partitions to resolve a download mismatch.

For the historical research commands, add `--research` to also populate that snapshot's expected Kaggle path. No field images enter primary fitting. An optional repeat of the actual cleaning/grouping implementation saves a separate audit:

```powershell
.\.venv\Scripts\python.exe tools\audit_dataset.py --output runs\image_audit_01
```

## C. Model training from scratch — optional, not executed during packaging

```powershell
.\.venv\Scripts\python.exe tools\train_primary.py --output runs\primary_refit_01 --check-only
.\.venv\Scripts\python.exe tools\train_primary.py --output runs\primary_refit_01
```

The first command only checks inputs/configuration. The second really retrains and can take substantial time. It clones each saved recipe to remove fitted state, then refits visual vocabulary, scaler, Nyström, LDA/PCA and estimator strictly inside each original grouped CV fitting fold. It reports mean fold CV scores, pooled CV class/confusion metrics and final train/validation scores, then writes newly fitted bundles to the requested new directory. It never trains on the internal test or field data. Exact serialized byte equality across machines is not promised. The application continues using the frozen release bundles until a separate measured promotion decision.

## D. Saved-model evaluation — no fitting

```powershell
.\.venv\Scripts\python.exe tools\evaluate_primary.py --output runs\validation_eval_01
.\.venv\Scripts\python.exe tools\evaluate_primary.py --partition both --output runs\development_eval_01
```

The first evaluates validation only; `both` adds resubstitution train metrics. Expected accuracy and F1 are in MODEL_RESULTS.md. Output is `metrics.json`, including per-class precision/recall/F1 and raw confusion counts in fixed class order. CV is not estimated by applying a full-training model to its training rows; use the stored grouped CV evidence or the real fold-refit training command. Neither default command accesses a test set. Every output directory must be new.

## E. Streamlit with saved models

```powershell
.\run_gui.bat
```

Equivalent:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

Open http://localhost:8501. Upload JPEG, PNG or WebP. Dataset examples are enabled after importing Kaggle. LR and KNN show five estimated probabilities; K-Means shows a training-mapped cluster label and cluster number. No fitting occurs in Streamlit. If the port is busy, append `--server.port 8502` to the equivalent command.

## Preprocessing and model mechanics

The exact original feature modules are retained in both app and primary code. EXIF orientation is corrected; images convert to RGB and use a 128×128 center crop. Handcrafted color histograms and spatial color descriptors, HOG, LBP, multi-scale edges and 98 local patch descriptors describe each photo. Twenty-four learned visual words pool signed local residuals. Standardization and group weights prevent dimensions alone from determining the representation. Check each bundle's saved pipeline and MODEL_MANIFEST.json for actual fitted transformations.

LR uses all five feature groups, a degree-2 polynomial Nyström approximation (gamma 0.18, 1,300 landmarks), and Logistic Regression C=3. KNN uses the same raw groups, a 500-landmark degree-2 map, four-dimensional supervised LDA, 31 Euclidean neighbors and uniform voting. K-Means uses local visual words, PCA20 and 10 clusters with k-means++, n_init=10 and random_state=42; fitting labels only name clusters after unsupervised centroid fitting. No pretrained models, deep learning or external embeddings are used.

## Partitions, leakage checks and interpretation

Frozen Kaggle partitions contain 3,017 fitting, 645 validation and 645 internal-test photos, with three grouped CV folds inside fitting only. Seed 42 is retained. Grouping detects exact bytes, decoded pixels, available Flickr photo IDs and perceptual similarities. All five labels occur in every required partition/fold. The rebuilt split has no detected group crossings; the old legacy split had eight, which is why it is archival. Observer grouping is retained separately for field experiments. Detection cannot rule out every edited copy or observer alias.

All published test cohorts were previously accessed, including the rebuilt internal cohort's reuse of legacy development photos. They are not new untouched benchmarks. Repeatedly accessed CV/validation results remain development estimates. Existing learning curves concern earlier fitted recipes, not new final-model learning-curve experiments. See MISSING_ARTIFACTS.md for exact limitations, including end-to-end LOO and academic eligibility.

## Inspecting and repeating history

Original scripts are preserved without path or hyperparameter edits in the research snapshot. Use TRAINING_GUIDE.md to make a writable copy and rerun the original optimization stages. All 124 rows remain in its `part_b_images/rebuild/final_optimization/ALL_EXPERIMENTS.csv`; primary-only rows are also filtered into `primary/results/ALL_EXPERIMENTS.csv`. The initial selection policy and documented development-only stability amendment remain available, including rejected candidates and backups.

## Team testing and bugs

After dataset import, run `.venv\Scripts\python.exe app\tests\verify_release.py` for all 15 original-inference fixture checks. Follow `app/tests/README.md` for browser checks. Root `tools/verify_package.py` checks models, manifests, research records and copied hashes even before dataset import. Add `--check-dataset` after import.

Use `app/BUG_REPORT_TEMPLATE.md` for upload problems, crashes or wrong classifications. Include release name/date, model, image/source, expected label with identification basis, actual label/probabilities/cluster, reproduction steps, screenshot and terminal traceback. Keep newly supplied team photos in a separate review set; do not overwrite frozen models or put test images into training. No oral-examination PDF was created.
