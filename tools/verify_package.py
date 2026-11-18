"""Check required artifacts, source hashes, primary splits and frozen model recipes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from common import ROOT, FILES, CODE, development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-dataset", action="store_true")
    args = parser.parse_args()
    for name in ("README.md", "FILE_STRUCTURE.md", "TRAINING_GUIDE.md", "MODEL_RESULTS.md",
                 "MISSING_ARTIFACTS.md", "TEAM_HANDOVER.md", "requirements-lock.txt"):
        assert (ROOT / name).is_file(), name
    records = json.loads((ROOT / "verification/COPY_MANIFEST.json").read_text())
    # app.py has a documented missing-dataset example-button guard; every other copied payload stays exact.
    checked = 0
    for item in records:
        path = ROOT / item["destination"]
        if item["destination"] == "app/app.py":
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
        checked += 1
    frozen = json.loads((ROOT / "app/MODEL_MANIFEST.json").read_text())
    for item in frozen:
        assert item["fit_counts"] == {"kaggle": 3017}
        model = ROOT / "app" / item["path"]
        assert hashlib.sha256(model.read_bytes()).hexdigest() == item["sha256"]
        bundle = joblib.load(model)
        clean = clone(bundle["pipeline"])
        assert clean.get_params(deep=False)["memory"] == bundle["pipeline"].get_params(deep=False)["memory"]
        assert not hasattr(clean.named_steps["estimator"], "classes_")
        if item["family"] == "Logistic Regression":
            assert clean.named_steps["estimator"].C == 3
            assert clean.named_steps["similarities"].n_components == 1300
        elif item["family"] == "KNN Classifier":
            estimator = clean.named_steps["estimator"]
            assert (estimator.n_neighbors, estimator.metric, estimator.weights) == (31, "euclidean", "uniform")
            assert clean.named_steps["similarities"].n_components == 500
        else:
            assert clean.named_steps["estimator"].n_clusters == 10
            assert clean.named_steps["pca"].n_components == 20
    rows, fit, held, folds = development()
    manifest = pd.read_csv(CODE / "results/manifest.csv")
    assert manifest.groupby("group").split.nunique().max() == 1
    assert manifest.groupby("split").label.nunique().eq(5).all()
    assert manifest.split.value_counts().to_dict() == {"train": 3017, "validation": 645, "test": 645}
    original = pd.read_csv(CODE / "results/cv_folds.csv").set_index("path")
    assert np.array_equal(rows.iloc[fit].fold, original.loc[rows.path.iloc[fit], "fold"])
    research = ROOT / "supplemental_field/research_project/part_b_images/rebuild"
    experiments = pd.read_csv(research / "final_optimization/ALL_EXPERIMENTS.csv")
    assert len(experiments) == 124 and set(experiments.scope) == {"primary", "supplemental"}
    assert pd.read_csv(ROOT / "primary/results/ALL_EXPERIMENTS.csv").scope.eq("primary").all()
    field = pd.read_csv(research / "generalization/field_manifest.csv")
    dev = pd.read_csv(research / "generalization/development_split.csv")
    field = field.merge(dev[dev.source.eq("field")][["path", "split", "fold"]], on="path", validate="one_to_one")
    assert field.groupby("observer").split.nunique().max() == 1
    assert field.groupby("observer").fold.nunique().max() == 1
    assert field.split.value_counts().to_dict() == {"train": 450, "validation": 112}
    cohorts = [field, *[pd.read_csv(research / path) for path in (
        "generalization/fresh_manifest.csv", "knn_study/fresh_manifest.csv",
        "logistic_study/fresh_manifest.csv", "results/external_manifest.csv")]]
    for index, a in enumerate(cohorts):
        for b in cohorts[index + 1:]:
            assert not set(a.observer) & set(b.observer)
            assert not set(a.sha256) & set(b.sha256)
    assert not set(manifest.sha256) & set(pd.concat(cohorts).sha256)
    for folder in (research / "generalization", research / "knn_study", research / "logistic_study"):
        for field in folder.glob("*manifest.csv"):
            data = pd.read_csv(field)
            for record in data.to_dict("records"):
                path = ROOT / "supplemental_field/research_project" / record["path"]
                assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    verified_images = 0
    if args.check_dataset:
        inventory = pd.read_csv(CODE / "results/raw_inventory.csv")
        for row in inventory.itertuples():
            path = ROOT / "primary" / row.path
            assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256
        verified_images = len(inventory)
    result = {"result": "passed", "copied_file_hashes_verified": checked,
              "finalized_model_hashes_verified": 3, "final_experiments": len(experiments),
              "primary_training": len(fit), "primary_validation": len(held), "primary_internal_test": 645,
              "grouped_cv_folds": len(folds), "raw_dataset_images_verified": verified_images,
              "field_observer_and_cohort_separation": "passed",
              "training_executed": False, "clone_reset_verified": True}
    (ROOT / "verification/PACKAGE_CHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
