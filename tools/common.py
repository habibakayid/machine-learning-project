"""Shared access to frozen primary partitions and the original feature extractor."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "primary"
CODE = PRIMARY / "part_b_images/rebuild"
sys.path.insert(0, str(CODE))
from dataset_preparation import CLASSES
from feature_extraction import cached_features
from train_models import combine

FILES = {"Logistic Regression": "logistic_regression.joblib",
         "KNN Classifier": "knn_classifier.joblib", "K-Means": "kmeans.joblib"}


def development():
    rows = pd.read_csv(PRIMARY / "development_split.csv")
    inventory = pd.read_csv(CODE / "results/manifest.csv").set_index("path")
    rows["sha256"] = rows.path.map(inventory.sha256)
    assert rows.source.eq("kaggle").all() and rows.path.is_unique
    assert rows.sha256.notna().all()
    assert rows.groupby("group").split.nunique().max() == 1
    assert rows.groupby("group").fold.nunique().max() == 1
    assert rows.groupby("split").label.nunique().eq(5).all()
    fit = np.flatnonzero(rows.split.eq("train"))
    held = np.flatnonzero(rows.split.eq("validation"))
    assert len(fit) == 3017 and len(held) == 645
    folds = [(fit[rows.fold.iloc[fit].ne(i)], fit[rows.fold.iloc[fit].eq(i)]) for i in (1, 2, 3)]
    for a, b in folds:
        assert not set(rows.group.iloc[a]) & set(rows.group.iloc[b])
    return rows, fit, held, folds


def features(rows, bundle):
    experiment = bundle["experiment"]
    # The three selected recipes all use the original crop extractor.
    assert experiment["mode"] == "crop"
    parts = cached_features(rows, "crop")
    if "local" in experiment["groups"]:
        parts.update(cached_features(rows, "local-crop"))
    return combine(parts, experiment["groups"])


def new_output(path):
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output
