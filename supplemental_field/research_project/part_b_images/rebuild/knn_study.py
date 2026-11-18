"""Tune KNN on the existing grouped folds and score fixed models on a new field set."""

import argparse
from dataclasses import asdict, dataclass, replace
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from dataset_preparation import CLASSES, RESULTS, SEED
from generalization_data import FIELD
from generalization_study import FOREGROUND, OUTPUT as GENERAL, STUDY_MODELS, export_metrics, parts_for
from knn_holdout import HOLDOUT, OUTPUT
from train_models import FILES, Experiment, accuracy_and_f1, combine, make_pipeline, predict

NEIGHBORS = (3, 5, 7, 11, 15, 21)
METRICS = ("euclidean", "manhattan", "cosine")
VOTES = ("uniform", "distance")
SOURCES = ("kaggle", "field")
ORIGINAL = STUDY_MODELS / FILES["KNN Classifier"]
MODEL = OUTPUT / "knn_classifier.joblib"
LABELS = np.asarray(CLASSES)


@dataclass(frozen=True)
class Geometry:
    features: Experiment
    projector: str = ""
    fit_balance: bool = False
    pool: str = "all"


def development_rows():
    dev = pd.read_csv(GENERAL / "development_split.csv")
    hashes = pd.concat([pd.read_csv(RESULTS / "manifest.csv"), pd.read_csv(FIELD)])
    dev["sha256"] = dev.path.map(hashes.set_index("path").sha256)
    assert dev.sha256.notna().all() and dev.path.is_unique
    assert dev.groupby("group").split.nunique().max() == 1
    assert dev.groupby("group").fold.nunique().max() == 1
    assert dev[dev.split == "train"].groupby(["source", "fold"]).label.nunique().eq(5).all()
    return dev


def groups_for(dev, experiment):
    parts = [parts_for(dev[dev.source == source], experiment) for source in SOURCES]
    # The original manifest is ordered Kaggle first, followed by field photos.
    assert dev.source.tolist() == sorted(dev.source, key=SOURCES.index)
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def transformations(spec, dimensions):
    steps = make_pipeline(spec.features, dimensions).steps[:-1]
    if spec.projector == "selection":
        steps.extend([("varying", VarianceThreshold()), ("selected", SelectKBest(k=384))])
    if spec.projector == "lda":
        # LDA supplies four label-aware coordinates. KNN still makes the final vote.
        steps.append(("projection", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")))
    return Pipeline(steps)


def fit_space(spec, dimensions, x, labels, sources, idx):
    transform = transformations(spec, dimensions)
    fitted = idx
    if spec.fit_balance:
        field = idx[sources[idx] == "field"]
        repeats = max(1, round(np.count_nonzero(sources[idx] == "kaggle") / len(field)))
        fitted = np.concatenate([idx, np.tile(field, repeats - 1)])
    transform.fit(x[fitted], labels[fitted])
    pool = idx
    if spec.pool == "denser field":
        rng = np.random.default_rng(SEED)
        keep = [idx[sources[idx] == "field"]]
        for label in CLASSES:
            kaggle = idx[(sources[idx] == "kaggle") & (labels[idx] == label)]
            field_count = np.count_nonzero(labels[keep[0]] == label)
            keep.append(rng.choice(kaggle, size=min(len(kaggle), 3 * field_count), replace=False))
        pool = np.sort(np.concatenate(keep))
    return transform, pool, np.ascontiguousarray(transform.transform(x[pool]))


def knn(metric, k, vote):
    return KNeighborsClassifier(n_neighbors=k, metric=metric, weights=vote, algorithm="brute")


def vote_labels(distance, neighbor_class, weighting):
    if weighting == "distance":
        weights = np.zeros_like(distance)
        np.divide(1, distance, out=weights, where=distance != 0)
        exact = (distance == 0).any(axis=1)
        weights[exact] = distance[exact] == 0
    else:
        weights = np.ones_like(distance)
    counts = np.stack([(weights * (neighbor_class == i)).sum(axis=1) for i in range(len(CLASSES))])
    return LABELS[counts.argmax(axis=0)]


def input_data(dev, spec, raw_cache):
    exp = spec.features
    key = exp.mode, "local" in exp.groups
    if key not in raw_cache:
        raw_cache[key] = groups_for(dev, exp)
    parts = raw_cache[key]
    return combine(parts, exp.groups), [parts[group].shape[1] for group in exp.groups]


def search_geometry(dev, spec, folds, raw_cache, stage, ks=NEIGHBORS, metrics=METRICS):
    x, sizes = input_data(dev, spec, raw_cache)
    labels = dev.label.to_numpy(dtype=str)
    source = dev.source.to_numpy(dtype=str)
    config = list(product(metrics, ks, VOTES))
    scores = {key: [] for key in config}
    for fit, held in folds:
        transform, pool, fit_x = fit_space(spec, sizes, x, labels, source, fit)
        held_x = np.ascontiguousarray(transform.transform(x[held]))
        ids = np.searchsorted(LABELS, labels[pool])
        masks = [source[held] == s for s in SOURCES]
        for metric in metrics:
            estimator = knn(metric, max(ks), "uniform").fit(fit_x, labels[pool])
            distances, neighbors = estimator.kneighbors(held_x)
            for k, vote in product(ks, VOTES):
                pred = vote_labels(distances[:, :k], ids[neighbors[:, :k]], vote)
                scores[metric, k, vote].append([
                    accuracy_and_f1(labels[held][mask], pred[mask]) for mask in masks
                ])
    rows = []
    exp = spec.features
    for (metric, k, vote), values in scores.items():
        mean = np.asarray(values).mean(axis=0)
        rows.append({
            "stage": stage, "representation": exp.name, "mode": exp.mode, "groups": exp.groups,
            "weights": exp.weights, "scale": exp.scale, "pca": exp.pca, "kernel": exp.kernel,
            "landmarks": exp.components if exp.kernel else 0, "projector": spec.projector,
            "fit_balance": spec.fit_balance, "neighbor_pool": spec.pool, "metric": metric, "k": k,
            "vote": vote, "kaggle_cv_accuracy": mean[0, 0], "kaggle_cv_macro_f1": mean[0, 1],
            "field_cv_accuracy": mean[1, 0], "field_cv_macro_f1": mean[1, 1],
            "selection_f1": mean[:, 1].mean(),
        })
    return rows


def feature_candidates(baseline):
    full = ("color", "hog", "lbp", "edges")
    compact = replace(baseline, name="Compact crop", mode="compact-crop", pca=128)
    global_features = replace(baseline, name="Global crop normalized", mode="crop", groups=full,
                              weights=(2., .6, .9, .8), scale="l2", pca=128)
    local = replace(global_features, name="Global and local normalized", groups=(*full, "local"),
                    weights=(2., .55, .7, .7, 1.25))
    return [
        replace(baseline, name="Compact full frame", pca=128),
        compact, replace(compact, name="Compact crop normalized", scale="l2"),
        replace(global_features, name="Joint color alone", groups=("color",), weights=(1.,)),
        global_features,
        replace(global_features, name="Global crop standardized", scale="standard"),
        local, replace(local, name="Global and local standardized", scale="standard"),
        replace(local, name="Global and local full frame", mode="pad"),
        replace(local, name="Color texture and local", groups=("color", "lbp", "local"),
                weights=(2., 1., 1.4)),
    ]


def report_config(dev, spec, row, folds, raw_cache, tag, archive=None):
    x, sizes = input_data(dev, spec, raw_cache)
    labels = dev.label.to_numpy(dtype=str)
    source = dev.source.to_numpy(dtype=str)
    indices = np.flatnonzero(dev.split == "train")
    validation = np.flatnonzero(dev.split == "validation")
    frames, neighborhoods = [], []

    def fitted(ids):
        transform, pool, fit_x = fit_space(spec, sizes, x, labels, source, ids)
        estimator = knn(row["metric"], row["k"], row["vote"]).fit(fit_x, labels[pool])
        pipeline = Pipeline([*transform.steps, ("estimator", estimator)])
        config = replace(spec.features, name=tag, metric=row["metric"], neighbors=row["k"],
                         vote=row["vote"])
        return {"pipeline": pipeline, "mapping": None, "experiment": asdict(config),
                "dimensions": sizes}, pool

    def record(bundle, ids, split):
        return dev.iloc[ids][["path", "source", "label"]].assign(
            candidate=tag, split=split, predicted=predict(bundle, x[ids])
        )

    for fold, (fit, held) in enumerate(folds, 1):
        model, pool = fitted(fit)
        frame = record(model, held, f"cv_{fold}")
        frames.append(frame)
        model_pipe = model["pipeline"]
        distances, nearest = model_pipe.named_steps["estimator"].kneighbors(
            model_pipe[:-1].transform(x[held]))
        nn = pool[nearest]
        neighborhoods.append(frame.assign(
            field_neighbor_fraction=(source[nn] == "field").mean(axis=1),
            same_class_fraction=(labels[nn] == labels[held, None]).mean(axis=1),
            nearest_distance=distances[:, 0],
        ))
    model, pool = fitted(indices)
    if archive is not None:
        model = archive
    frames.extend([record(model, indices, "train"), record(model, validation, "validation")])
    pipe = model["pipeline"]
    groups = dev.group.to_numpy(dtype=str)
    extra = pd.Series(groups[pool]).value_counts().max()
    fitted_neighbors = pipe.named_steps["estimator"]
    distances, nearest = fitted_neighbors.kneighbors(
        pipe[:-1].transform(x[indices]), n_neighbors=row["k"] + extra)
    distinct = groups[pool[nearest]] != groups[indices, None]
    keep = np.argsort(~distinct, axis=1, kind="stable")[:, :row["k"]]
    ungrouped = vote_labels(np.take_along_axis(distances, keep, axis=1),
                            np.searchsorted(LABELS, labels[pool[np.take_along_axis(nearest, keep, axis=1)]]),
                            row["vote"])
    frames.append(dev.iloc[indices][["path", "source", "label"]].assign(
        candidate=tag, split="train_without_group", predicted=ungrouped))
    print(f"Verified {tag} with the actual sklearn pipeline.", flush=True)
    return model, pd.concat(frames), pd.concat(neighborhoods)


def development():
    OUTPUT.mkdir(exist_ok=True)
    dev = development_rows()
    archived = joblib.load(ORIGINAL)
    baseline = Experiment(**archived["experiment"])
    train = np.flatnonzero(dev.split == "train")
    folds = [(train[dev.fold.iloc[train] != i], train[dev.fold.iloc[train] == i])
             for i in (1, 2, 3)]
    rows, tried, raw_cache = [], {}, {}

    def run(spec, stage, ks=NEIGHBORS, metrics=METRICS):
        if spec in tried:
            return tried[spec]
        table = search_geometry(dev, spec, folds, raw_cache, stage, ks, metrics)
        rows.extend(table)
        tried[spec] = max(table, key=lambda row: row["selection_f1"])
        pd.DataFrame(rows).sort_values("selection_f1", ascending=False).to_csv(
            OUTPUT / "experiments.csv", index=False
        )
        print(f"{stage} | {spec.features.name} | PCA {spec.features.pca} | {spec.projector or 'plain'}"
              f" | {tried[spec]['selection_f1']:.2%} CV Macro F1", flush=True)
        return tried[spec]

    features = [Geometry(exp) for exp in feature_candidates(baseline)]
    for spec in features:
        run(spec, "features", (5, 11, 15))
    shortlist = sorted(features, key=lambda spec: tried[spec]["selection_f1"], reverse=True)[:2]
    # Use the complete neighbor grid on the two best representations and all PCA sizes.
    for seed in shortlist:
        for size in (64, 128, 256, 0):
            spec = replace(seed, features=replace(seed.features, pca=size))
            if spec in tried:
                del tried[spec]
            run(spec, "PCA and neighbors")
    best_plain = max(tried, key=lambda spec: tried[spec]["selection_f1"])
    exp = best_plain.features
    for group in ("color", "local", "shape"):
        targets = {"color": ("color", "global_color", "foreground_color"),
                   "local": ("local",), "shape": ("hog", "lbp", "edges", "foreground_shape",
                                                  "global_shape")}[group]
        if any(key in targets for key in exp.groups):
            weights = tuple(weight * 1.6 if name in targets else weight
                            for name, weight in zip(exp.groups, exp.weights, strict=True))
            run(replace(best_plain, features=replace(exp, name=f"{exp.name}, heavier {group}",
                                                     weights=weights)), "group weights")
    run(replace(best_plain, pool="denser field"), "neighbor pool")

    # The optional supervised steps use labels from the fitting fold only.
    for seed in shortlist:
        config = replace(seed.features, pca=128, kernel="")
        run(Geometry(replace(config, pca=0), projector="selection"), "supervised selection")
        run(Geometry(config, projector="lda"), "supervised projection")
        run(Geometry(config, projector="lda", fit_balance=True), "supervised projection")
    rich = next(spec.features for spec in features if spec.features.name == "Global and local standardized")
    projected = replace(rich, name="Global and local polynomial", pca=0, kernel="poly",
                        gamma=.18, components=500)
    run(Geometry(projected, projector="lda"), "supervised projection")
    run(Geometry(projected, projector="lda", fit_balance=True), "supervised projection")

    winner = max(tried, key=lambda spec: tried[spec]["selection_f1"])
    chosen = tried[winner]
    baseline_row = {"metric": baseline.metric, "k": baseline.neighbors, "vote": baseline.vote}
    _, old, old_nn = report_config(dev, Geometry(baseline), baseline_row, folds, raw_cache,
                                    "Archived KNN", archive=archived)
    bundle, new, new_nn = report_config(dev, winner, chosen, folds, raw_cache, "Selected KNN")
    pd.DataFrame([chosen]).to_csv(OUTPUT / "selected.csv", index=False)
    joblib.dump(bundle, MODEL, compress=3)
    predictions = pd.concat([old, new], ignore_index=True)
    predictions.to_csv(OUTPUT / "development_predictions.csv", index=False)
    export_metrics(predictions, "development", OUTPUT)
    exact_cv = predictions[predictions.split.str.startswith("cv_")]
    verified = []
    for (candidate, source), frame in exact_cv.groupby(["candidate", "source"]):
        fold_scores = [accuracy_and_f1(rows.label, rows.predicted)
                       for _, rows in frame.groupby("split")]
        accuracy, macro_f1 = np.mean(fold_scores, axis=0)
        verified.append({"candidate": candidate, "source": source, "accuracy": accuracy,
                         "macro_f1": macro_f1})
    pd.DataFrame(verified).to_csv(OUTPUT / "verified_cv.csv", index=False)
    pd.concat([old_nn, new_nn]).groupby(
        ["candidate", "source", "label"], as_index=False
    )[["field_neighbor_fraction", "same_class_fraction", "nearest_distance"]].mean().to_csv(
        OUTPUT / "neighbor_audit.csv", index=False
    )


def final_test():
    rows = pd.read_csv(HOLDOUT).assign(source="new_field")
    results = []
    study_models = [(ORIGINAL, "Archived KNN"), (MODEL, "Selected KNN"),
                    (STUDY_MODELS / FILES["Logistic Regression"], "Frozen Logistic Regression")]
    for path, name in study_models:
        bundle = joblib.load(path)
        experiment = Experiment(**bundle["experiment"])
        x = combine(parts_for(rows, experiment), experiment.groups)
        frame = rows[["path", "source", "label"]].assign(
            candidate=name, split="fresh_test", predicted=predict(bundle, x)
        )
        print(name, accuracy_and_f1(frame.label, frame.predicted), flush=True)
        results.append(frame)
    output = pd.concat(results, ignore_index=True)
    output.to_csv(OUTPUT / "fresh_predictions.csv", index=False)
    export_metrics(output, "fresh", OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-test", action="store_true", help="Evaluate the fixed model and references.")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        (final_test if args.fresh_test else development)()
