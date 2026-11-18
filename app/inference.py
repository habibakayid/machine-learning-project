"""Inference only; function bodies copied verbatim from the finalized project."""
from pathlib import Path
import numpy as np
from dataset_preparation import CLASSES
from feature_extraction import extract
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
FILES = {'Logistic Regression': 'logistic_regression.joblib', 'KNN Classifier': 'knn_classifier.joblib', 'K-Means': 'kmeans.joblib'}

def combine(parts, groups):
    return np.hstack([parts[name] for name in groups])

def predict_image(bundle, image):
    experiment = bundle["experiment"]
    mode = experiment["mode"]
    if mode.startswith("compact-"):
        from generalization_features import extract_compact
        parts = extract_compact(image, mode)
    else:
        parts = extract(image, mode)
    if "local" in experiment["groups"]:
        parts.update(extract(image, f"local-{experiment['mode']}"))
    x = combine(parts, experiment["groups"]).reshape(1, -1)
    pipeline = bundle["pipeline"]
    if bundle["mapping"] is not None:
        cluster = pipeline.predict(x).item()
        return str(bundle["mapping"][cluster]), None, cluster
    probabilities = dict(zip(pipeline.classes_, pipeline.predict_proba(x)[0], strict=True))
    return str(pipeline.predict(x).item()), probabilities, None
