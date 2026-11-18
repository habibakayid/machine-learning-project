# Package structure and actual source locations

```text
FULL_PART_B_TEAM_RELEASE/
  README.md
  FILE_STRUCTURE.md
  TRAINING_GUIDE.md
  MODEL_RESULTS.md
  MISSING_ARTIFACTS.md
  TEAM_HANDOVER.md
  setup_windows.bat
  run_gui.bat
  requirements.txt
  requirements-lock.txt
  .streamlit/config.toml
  app/
    app.py, inference.py, dataset_preparation.py
    feature_extraction.py, local_features.py, generalization_features.py
    models/{logistic_regression,knn_classifier,kmeans}.joblib
    MODEL_MANIFEST.json
    samples/MANIFEST.json       # images restored by recipient's dataset import
    tests/                     # 15 fixture expectations, direct/browser checks
    evidence/                  # executed release checks
    BUG_REPORT_TEMPLATE.md
  primary/
    development_split.csv
    train_manifest.csv, validation_manifest.csv, test_manifest.csv
    original_splits/           # legacy clean/excluded/train/validation/test CSVs
    part_b_images/rebuild/     # original core feature/model/audit source
      results/                 # frozen inventory, groups, exclusions, CV folds
    part_b_images/data/raw/flowers/  # created only by dataset import
    results/                   # filtered primary-only final metrics/experiments
    figures/                   # final primary plots + labeled historical curve
  supplemental_field/
    PHOTO_ATTRIBUTION.csv
    research_project/          # preserved full research layout; mixed history
      models/                  # original legacy model files
      gui/                     # original GUI and backups
      part_b_images/scripts/   # all 20 original legacy scripts
      part_b_images/data/processed/ # numeric legacy feature files and split CSVs
      part_b_images/rebuild/
        *.py                   # all actual study and evaluation implementations
        models/, results/, figures/
        generalization/        # field training, grouped split, models, results
        logistic_study/, knn_study/, kmeans_study/
        final_optimization/    # complete 124-candidate studies and all backups
          primary/, supplemental/
        final_review/          # original technical audits and cohort qualifications
  datasets/LICENSE_STATUS.md
  docs/ORIGINAL_COURSE_GUIDELINES.pdf  # existing course document, not an examination PDF
  tools/                       # import, audit, verification, refit, evaluation
  verification/                # provenance, exclusions, checks, file inventory
```

The historical snapshot includes Kaggle baselines required by the supplemental comparisons; it is explicitly an archive of mixed research, not the primary results directory. Numerical tables inside it retain their original source/scope columns. No supplemental model is selected by app/app.py. Copy this snapshot using tools/research_workspace.py before rerunning mutation-capable scripts.

The generated verification/FILE_LIST.txt lists every distributed file. verification/FILE_INVENTORY.csv records payload paths, byte sizes and SHA256 values, excluding the inventory and checksum index themselves. verification/COPY_MANIFEST.json provides source provenance. Environments, feature caches, raw Kaggle photos/pixel arrays and unlicensed photo previews are intentionally not part of the ZIP. No symbolic links or machine-specific absolute paths are required by the supported entry points.
