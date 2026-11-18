"""Compare handcrafted features with group-aware CV and save one model per family."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from threadpoolctl import threadpool_limits

from dataset_preparation import CLASSES, RESULTS, ROOT, SEED
from feature_extraction import cached_features, extract
from local_features import DESCRIPTOR_SIZE, LocalVisualWords

MODELS = Path(__file__).resolve().parent / "models"
FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "KNN Classifier": "knn_classifier.joblib",
    "K-Means": "kmeans.joblib",
}
ALL = ("color", "hog", "lbp", "edges")
LEGACY = ("legacy_hog", "rgb", "legacy_hsv")


@dataclass(frozen=True)
class Experiment:
    name: str
    model: str = "Logistic Regression"
    mode: str = "pad"
    groups: tuple = ALL
    weights: tuple = (2., .65, .85, .85)
    scale: str = "l2"
    c: float = 10.
    pca: int = 0
    kernel: str = ""
    gamma: float = .45
    components: int = 700
    augment: tuple = ()
    neighbors: int = 11
    metric: str = "euclidean"
    vote: str = "distance"
    n_init: int = 15
    local_words: int = 24
    one_to_one: bool = False


def combine(parts, groups):
    return np.hstack([parts[name] for name in groups])


def make_pipeline(experiment, dimensions):
    start = 0
    transforms, weights = [], {}
    for name, size, weight in zip(experiment.groups, dimensions, experiment.weights, strict=True):
        standardize = experiment.scale == "standard"
        normalizer = StandardScaler() if standardize else Normalizer()
        if name == "local":
            normalizer = LocalVisualWords(n_words=experiment.local_words)
        transforms.append((name, normalizer, slice(start, start + size)))
        weights[name] = weight / np.sqrt(size) if standardize and name != "local" else weight
        start += size
    steps = [("feature_groups", ColumnTransformer(transforms, transformer_weights=weights))]
    if experiment.pca:
        steps.append(("pca", PCA(experiment.pca, svd_solver="randomized", random_state=SEED)))
    if experiment.kernel:
        steps.append(("similarities", Nystroem(
            kernel=experiment.kernel, gamma=experiment.gamma, degree=2,
            n_components=experiment.components, random_state=SEED,
        )))

    if experiment.model == "KNN Classifier":
        estimator = KNeighborsClassifier(
            n_neighbors=experiment.neighbors, metric=experiment.metric, weights=experiment.vote
        )
    elif experiment.model == "K-Means":
        estimator = KMeans(n_clusters=len(CLASSES), n_init=experiment.n_init, random_state=SEED)
    else:
        estimator = LogisticRegression(C=experiment.c, max_iter=1000, random_state=SEED)
    return Pipeline([*steps, ("estimator", estimator)])


def fit_model(experiment, dimensions, x, y, sample_weight=None):
    fit_options = {} if sample_weight is None else {"estimator__sample_weight": sample_weight}
    pipeline = make_pipeline(experiment, dimensions).fit(x, y, **fit_options)
    mapping = None
    if experiment.model == "K-Means":
        # Use training labels to name clusters. The one-to-one version gives
        # each of the five classes an assigned cluster.
        assigned = pipeline.named_steps["estimator"].labels_
        labels = np.searchsorted(CLASSES, y)
        counts = confusion_matrix(assigned, labels, labels=np.arange(len(CLASSES)))
        cluster_labels = counts.argmax(axis=1)
        if experiment.one_to_one:
            clusters, cluster_labels = linear_sum_assignment(counts, maximize=True)
            assert np.array_equal(clusters, np.arange(len(CLASSES)))
        mapping = np.asarray(CLASSES)[cluster_labels]
    return {"pipeline": pipeline, "mapping": mapping, "experiment": asdict(experiment),
            "dimensions": list(dimensions)}


def predict(bundle, x):
    predictions = bundle["pipeline"].predict(x)
    return predictions if bundle["mapping"] is None else bundle["mapping"][predictions]


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


def accuracy_and_f1(y, pred):
    return accuracy_score(y, pred), f1_score(y, pred, labels=CLASSES, average="macro",
                                           zero_division=0)


def compare(experiment, parts, augmented, dev, train, validation, folds):
    dimensions = [parts[experiment.mode][name].shape[1] for name in experiment.groups]
    x = combine(parts[experiment.mode], experiment.groups)
    x_aug = [combine(augmented[experiment.mode, method], experiment.groups)
             for method in experiment.augment]
    y = dev.label.to_numpy(dtype=str)

    def fit(indices):
        return fit_model(experiment, dimensions, np.vstack([
            x[indices], *[extra[indices] for extra in x_aug]
        ]), np.tile(y[indices], 1 + len(x_aug)))

    cv_pred = y.copy()
    cv_scores = []
    for fit_idx, fold_idx in folds:
        cv_pred[fold_idx] = predict(fit(fit_idx), x[fold_idx])
        cv_scores.append(accuracy_and_f1(y[fold_idx], cv_pred[fold_idx]))

    bundle = fit(train)
    train_pred = predict(bundle, x[train])
    val_pred = predict(bundle, x[validation])
    train_accuracy, train_f1 = accuracy_and_f1(y[train], train_pred)
    val_accuracy, val_f1 = accuracy_and_f1(y[validation], val_pred)
    cv_accuracy, cv_f1 = np.mean(cv_scores, axis=0)
    descriptor_dimensions = sum(experiment.local_words * DESCRIPTOR_SIZE if name == "local" else size
                                for name, size in zip(experiment.groups, dimensions, strict=True))
    row = {**asdict(experiment), "input_dimensions": sum(dimensions),
           "dimensions": descriptor_dimensions,
           "estimator_dimensions": bundle["pipeline"].named_steps["estimator"].n_features_in_,
           "train_accuracy": train_accuracy, "train_macro_f1": train_f1,
           "cv_accuracy": cv_accuracy, "cv_accuracy_std": np.std(cv_scores, axis=0)[0],
           "cv_macro_f1": cv_f1, "validation_accuracy": val_accuracy,
           "validation_macro_f1": val_f1}
    if experiment.model == "K-Means":
        pipeline = bundle["pipeline"]
        coords = pipeline[:-1].transform(x[train])
        row["train_silhouette"] = silhouette_score(coords, pipeline.predict(x[train]),
                                                  sample_size=1300, random_state=SEED)
    prediction_frames = [
        dev.iloc[indices][["path", "group", "label"]].assign(
            candidate=experiment.name, evaluation=split, predicted=pred
        )
        for indices, pred, split in ((train, train_pred, "train"), (train, cv_pred[train], "cv"),
                                     (validation, val_pred, "validation"))
    ]
    print(f"{experiment.name:27s} train={train_accuracy:.2%} "
          f"CV={cv_accuracy:.2%} CV macro F1={cv_f1:.2%} val={val_accuracy:.2%}", flush=True)
    return row, bundle, pd.concat(prediction_frames, ignore_index=True)


def reproduce_legacy():
    """Refit the best recorded, unaugmented legacy configuration on its original split."""
    processed = ROOT / "part_b_images" / "data" / "processed"
    experiment = Experiment("Legacy original split", mode="legacy", groups=LEGACY,
                            weights=(1., 1., 1.), scale="standard", c=1.)
    def load(split):
        files = [f"features/{split}_hog.npz", f"features/{split}_color.npz",
                 f"features_spatial/{split}_spatial_hsv.npz"]
        labels, parts = [], []
        for file in files:
            with np.load(processed / file, allow_pickle=False) as saved:
                parts.append(saved["X"])
                labels.append(saved["y"])
        if any(not np.array_equal(labels[0], other) for other in labels[1:]):
            raise ValueError("Legacy cached labels disagree.")
        return np.hstack(parts), labels[0].astype(str), [part.shape[1] for part in parts]

    try:
        x, y, sizes = load("train")
        val_x, val_y, _ = load("validation")
    except FileNotFoundError:
        print("Original cached arrays unavailable. Skipping legacy reproduction.", flush=True)
        return
    legacy = fit_model(experiment, sizes, x, y)
    accuracy, macro_f1 = accuracy_and_f1(val_y, predict(legacy, val_x))
    pd.DataFrame([{"validation_accuracy": accuracy, "validation_macro_f1": macro_f1,
                   "recorded_validation_accuracy": .5959752321981424,
                   "reported_augmented_validation_accuracy": .6084}]).to_csv(
        RESULTS / "legacy_reproduction.csv", index=False
    )
    print(f"Original split reproduction: {accuracy:.2%} validation", flush=True)


def train_models():
    manifest = pd.read_csv(RESULTS / "manifest.csv")
    dev = manifest[manifest.split.isin(["train", "validation"])].reset_index(drop=True)
    train = np.flatnonzero(dev.split == "train")
    validation = np.flatnonzero(dev.split == "validation")
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = [(train[fit], train[held]) for fit, held in
             splitter.split(train, dev.label.iloc[train], dev.group.iloc[train])]
    fold_frames = []
    for fold, (fit, held) in enumerate(folds, 1):
        assert not set(dev.group.iloc[fit]) & set(dev.group.iloc[held])
        fold_frames.append(dev.iloc[held][["path", "label", "group"]].assign(fold=fold))
    pd.concat(fold_frames, ignore_index=True).to_csv(RESULTS / "cv_folds.csv", index=False)
    parts = {}
    augmented, rows, predictions, best = {}, [], [], {}
    MODELS.mkdir(exist_ok=True)

    def run(experiment):
        key = experiment.mode
        if key not in parts:
            parts[key] = cached_features(dev, key)
        if "local" in experiment.groups and "local" not in parts[key]:
            parts[key].update(cached_features(dev, f"local-{key}"))
        for method in experiment.augment:
            key = experiment.mode, method
            stored = augmented.setdefault(key, {})
            new_parts = {} if stored else cached_features(dev.iloc[train], *key)
            if "local" in experiment.groups and "local" not in stored:
                new_parts.update(cached_features(dev.iloc[train], f"local-{key[0]}", method))
            # Match the row indices of the original features. Only training rows are filled.
            for name, values in new_parts.items():
                aligned = np.zeros((len(dev), values.shape[1]), dtype=np.float32)
                aligned[train] = values
                stored[name] = aligned
        row, model, pred = compare(experiment, parts, augmented, dev, train, validation, folds)
        rows.append(row)
        predictions.append(pred)
        score = row["cv_macro_f1"], row["cv_accuracy"]
        family = experiment.model
        previous = best.get(family)
        if family == "K-Means":
            score = (experiment.one_to_one, *score)
        if previous is None or score > previous[0]:
            best[family] = score, model, row, experiment
        pd.DataFrame(rows).to_csv(RESULTS / "experiments.csv", index=False)

    reproduce_legacy()
    run_global_experiments(run, best)
    run_local_experiments(run, best)

    selected = []
    for family, (_, bundle, row, _) in best.items():
        joblib.dump(bundle, MODELS / FILES[family], compress=3)
        selected.append(row)
    pd.DataFrame(selected).sort_values(["cv_macro_f1", "cv_accuracy"], ascending=False).to_csv(
        RESULTS / "selected_models.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(RESULTS / "development_predictions.csv",
                                                     index=False)


def run_global_experiments(run, best):
    baseline = Experiment("Legacy HOG + RGB + HSV", mode="legacy", groups=LEGACY,
                          weights=(1., 1., 1.), scale="standard", c=1.)
    run(baseline)
    run(replace(baseline, name="Legacy + augmentation", augment=("flip", "mild")))
    for experiment in (
        Experiment("Joint color only", groups=("color",), weights=(2.,)),
        Experiment("HOG only", groups=("hog",), weights=(1.,)),
        Experiment("Color + HOG", groups=("color", "hog"), weights=(2., .65)),
        Experiment("Color + HOG + LBP", groups=("color", "hog", "lbp"), weights=(2., .65, .85)),
        Experiment("Color + LBP + edges", groups=("color", "lbp", "edges"), weights=(2., .85, .85)),
        Experiment("All features, pad"),
        Experiment("All features, crop", mode="crop"),
    ):
        run(experiment)
    base = best["Logistic Regression"][3]
    for experiment in (
        replace(base, name="Less regularization", c=80.),
        replace(base, name="Standardized groups", scale="standard", c=8.),
        replace(base, name="PCA, 160 dimensions", pca=160, c=40.),
        replace(base, name="RBF similarities", kernel="rbf", c=40.),
        replace(base, name="Quadratic similarities", kernel="poly", gamma=.25, c=15.),
    ):
        run(experiment)
    base = best["Logistic Regression"][3]
    run(replace(base, name="Flip augmentation", augment=("flip",)))
    run(replace(base, name="Flip + mild augmentation", augment=("flip", "mild")))
    if base.kernel == "rbf":
        run(replace(base, name="RBF, 1200 examples", components=1200, gamma=.65, c=80.,
                    augment=best["Logistic Regression"][3].augment))

    mode = base.mode if base.mode != "legacy" else "pad"
    for experiment in (
        Experiment("KNN 7 Euclidean uniform", model="KNN Classifier", mode=mode,
                   neighbors=7, vote="uniform"),
        Experiment("KNN 11 Euclidean distance", model="KNN Classifier", mode=mode),
        Experiment("KNN 21 with PCA", model="KNN Classifier", mode=mode, neighbors=21, pca=96),
        Experiment("KNN 11 Manhattan PCA", model="KNN Classifier", mode=mode,
                   metric="manhattan", pca=96),
        Experiment("KNN color and texture", model="KNN Classifier", mode=mode, neighbors=15,
                   groups=("color", "lbp", "edges"), weights=(2., 1., 1.)),
        Experiment("K-means all features", model="K-Means", mode=mode, pca=40),
        Experiment("K-means color and texture", model="K-Means", mode=mode, pca=24, n_init=30,
                   groups=("color", "lbp", "edges"), weights=(2., 1., 1.)),
        Experiment("K-means color only", model="K-Means", mode=mode, pca=16, n_init=30,
                   groups=("color",), weights=(1.,)),
    ):
        run(experiment)


def run_local_experiments(run, best):
    base = best["Logistic Regression"][3]
    mode = base.mode
    run(Experiment("Local visual words only", mode=mode, groups=("local",), weights=(1.,)))
    hybrid = Experiment("Global + local descriptors", mode=mode, groups=(*ALL, "local"),
                        weights=(2., .55, .7, .7, 1.25), scale="standard", c=8.)
    run(hybrid)
    run(replace(hybrid, name="Local replaces spatial HOG",
                groups=("color", "lbp", "edges", "local"), weights=(2.2, .7, .7, 1.65)))
    run(replace(hybrid, name="Global + local RBF", scale="l2", c=50., kernel="rbf",
                gamma=.3, components=1000))
    run(replace(hybrid, name="Global + local flip", augment=("flip",)))
    run(replace(hybrid, name="Global + local quadratic", c=18., kernel="poly",
                gamma=.18, components=1300))
    run(replace(hybrid, name="KNN with local descriptors", model="KNN Classifier",
                c=10., neighbors=13, scale="l2", pca=128))
    run(Experiment("KNN standardized and PCA", model="KNN Classifier", mode=mode,
                   neighbors=13, metric="manhattan", pca=96, scale="standard"))
    for experiment in (
        Experiment("K-Means all, one-to-one", model="K-Means", mode=mode, pca=40, one_to_one=True),
        Experiment("K-Means color, one-to-one", model="K-Means", mode=mode, pca=16, n_init=30,
                   groups=("color",), weights=(1.,), one_to_one=True),
        replace(hybrid, name="K-Means local, one-to-one", model="K-Means", pca=32,
                n_init=20, one_to_one=True),
    ):
        run(experiment)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        train_models()
