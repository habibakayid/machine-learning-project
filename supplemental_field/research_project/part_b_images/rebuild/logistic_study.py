"""Staged Logistic Regression experiments on the fixed Kaggle and field folds."""

import argparse
from dataclasses import asdict, replace

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from dataset_preparation import CLASSES, RESULTS, SEED
from generalization_data import FIELD
from generalization_study import OUTPUT as GENERAL, STUDY_MODELS, export_metrics, parts_for, score
from logistic_holdout import HOLDOUT, OUTPUT
from train_models import FILES, Experiment, combine, make_pipeline, predict

CURRENT = STUDY_MODELS / FILES["Logistic Regression"]
COMPACT = STUDY_MODELS / "compact_logistic.joblib"
MODEL = OUTPUT / "regularized_logistic.joblib"
SOURCES = ("kaggle", "field")
C_VALUES = (.03, .1, .3, 1., 3., 10., 18.)
BASELINE_NAME = "Current Nystrom"
COMPACT_NAME = "Compact benchmark"


def development_rows():
    dev = pd.read_csv(GENERAL / "development_split.csv")
    hashes = pd.concat([pd.read_csv(RESULTS / "manifest.csv"), pd.read_csv(FIELD)])
    dev["sha256"] = dev.path.map(hashes.set_index("path").sha256)
    assert dev.path.is_unique and dev.sha256.notna().all()
    assert set(dev.split) == {"train", "validation"}
    assert dev.groupby("group").split.nunique().max() == 1
    assert dev.groupby("group").fold.nunique().max() == 1
    assert dev[dev.split == "train"].groupby(["source", "fold"]).label.nunique().eq(5).all()
    assert len(dev[(dev.source == "field") & (dev.split == "train")]) == 450
    return dev


def features(dev, experiment):
    assert dev.source.tolist() == sorted(dev.source, key=SOURCES.index)
    result = [parts_for(dev[dev.source == source], experiment) for source in SOURCES]
    return {name: np.concatenate([parts[name] for parts in result]) for name in result[0]}


def fit_weights(rows, mode):
    if mode == "mixed":
        return None
    strata = rows.source if mode == "balanced" else rows.source + ":" + rows.label
    counts = strata.value_counts()
    return strata.map(len(rows) / (len(counts) * counts)).to_numpy()


def summarize(predictions):
    row = {}
    for source in SOURCES:
        subset = predictions[predictions.source == source]
        for split in ("train", "cv", "validation"):
            part = subset[subset.split.str.startswith(split)]
            metrics = pd.DataFrame([score(group) for _, group in part.groupby("split")])
            for metric in ("accuracy", "macro_f1"):
                row[f"{source}_{split}_{metric}"] = metrics[metric].mean()
        row[f"{source}_gap"] = row[f"{source}_train_accuracy"] - row[f"{source}_validation_accuracy"]
    row["selection_f1"] = np.mean([row[f"{source}_cv_macro_f1"] for source in SOURCES])
    return row


def compare(dev, raw, geometry, variants, folds):
    """Reuse the same fold-fitted features across C and sample-weight settings."""
    dimensions = [raw[name].shape[1] for name in geometry.groups]
    x = combine(raw, geometry.groups)
    y = dev.label.to_numpy(dtype=str)
    train = np.flatnonzero(dev.split == "train")
    val = np.flatnonzero(dev.split == "validation")
    results = {name: [] for name, _, _ in variants}
    bundles = {}

    def frame(ids, name, split, predicted):
        return dev.iloc[ids][["path", "source", "label"]].assign(
            candidate=name, split=split, predicted=predicted
        )

    for fold, (fit, held) in enumerate([*folds, (train, val)], 1):
        final = fold == 4
        transform = make_pipeline(geometry, dimensions)[:-1]
        fit_x = transform.fit_transform(x[fit], y[fit])
        held_x = transform.transform(x[held])
        for name, c, training in variants:
            estimator = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
            estimator.fit(fit_x, y[fit], sample_weight=fit_weights(dev.iloc[fit], training))
            if final:
                bundle = {"pipeline": Pipeline([*transform.steps, ("estimator", estimator)]),
                          "mapping": None, "dimensions": dimensions,
                          "experiment": asdict(replace(geometry, name=name, c=c))}
                bundles[name] = bundle
                results[name].append(frame(fit, name, "train", estimator.predict(fit_x)))
            split = "validation" if final else f"cv_{fold}"
            results[name].append(frame(held, name, split, estimator.predict(held_x)))

    for name, c, training in variants:
        predictions = pd.concat(results[name], ignore_index=True)
        yield {
            "name": name, "c": c, "training": training, "preprocessing": geometry.mode,
            "groups": geometry.groups, "weights": geometry.weights, "scale": geometry.scale,
            "kernel": geometry.kernel or "linear", "gamma": geometry.gamma if geometry.kernel else None,
            "landmarks": geometry.components if geometry.kernel else 0,
            "estimator_dimensions": bundles[name]["pipeline"].named_steps["estimator"].n_features_in_,
            **summarize(predictions),
        }, bundles[name], predictions


def confusion_plot(predictions, names, split, sources, filename):
    fig, axes = plt.subplots(len(sources), len(names), squeeze=False,
                             figsize=(4 * len(names), 4.2 * len(sources)), layout="constrained")
    for i, source in enumerate(sources):
        for j, name in enumerate(names):
            subset = predictions[(predictions.candidate == name) & (predictions.source == source)
                                 & predictions.split.str.startswith(split)]
            counts = confusion_matrix(subset.label, subset.predicted, labels=CLASSES)
            percentages = 100 * counts / np.maximum(1, counts.sum(axis=1, keepdims=True))
            ax = axes[i, j]
            ConfusionMatrixDisplay(percentages, display_labels=[c.title() for c in CLASSES]).plot(
                ax=ax, colorbar=False, cmap="Greens", values_format=".0f", xticks_rotation=28
            )
            ax.images[0].set_clim(0, 100)
            ax.set_title(f"{source.title()} · {name}")
    fig.suptitle("Confusion matrices, percentage of each true class")
    fig.savefig(OUTPUT / filename, dpi=135)
    plt.close(fig)


def development():
    OUTPUT.mkdir(exist_ok=True)
    dev = development_rows()
    archived = joblib.load(CURRENT)
    baseline = Experiment(**archived["experiment"])
    compact = Experiment(**joblib.load(COMPACT)["experiment"])
    train = np.flatnonzero(dev.split == "train")
    folds = [(train[dev.fold.iloc[train] != i], train[dev.fold.iloc[train] == i])
             for i in (1, 2, 3)]
    raw = features(dev, baseline)
    rows, predictions, settings = [], [], {}
    best = None

    def run(geometry, variants, stage, available=raw):
        nonlocal best
        for row, bundle, output in compare(dev, available, geometry, variants, folds):
            row["stage"] = stage
            name = row["name"]
            rows.append(row)
            predictions.append(output)
            settings[name] = geometry
            if name not in (BASELINE_NAME, COMPACT_NAME) and (
                best is None or row["selection_f1"] > best[0]["selection_f1"]
            ):
                best = row, bundle
            print(f"{name} | Kaggle CV F1 {row['kaggle_cv_macro_f1']:.2%} | "
                  f"field CV F1 {row['field_cv_macro_f1']:.2%} | mean {row['selection_f1']:.2%}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT / "experiments.csv", index=False)

    run(baseline, [(BASELINE_NAME if c == baseline.c else f"C={c:g}", c, "mixed")
                   for c in C_VALUES], "regularization")
    original = next(row for row in rows if row["name"] == BASELINE_NAME)
    regularized_c = best[0]["c"]
    c_values = list(dict.fromkeys([
        max(rows, key=lambda row: row["selection_f1"])["c"], regularized_c,
    ]))
    for size in (300, 600, 900):
        run(replace(baseline, components=size),
            [(f"{size} landmarks C={c:g}", c, "mixed") for c in c_values], "landmarks")

    leader = max(rows, key=lambda row: row["selection_f1"])
    anchor = settings[leader["name"]]
    c_idx = C_VALUES.index(leader["c"])
    nearby = C_VALUES[max(0, c_idx - 1):c_idx + 2]
    for gamma in (.03, .08):
        run(replace(anchor, gamma=gamma),
            [(f"Gamma {gamma:g} C={c:g}", c, "mixed") for c in nearby], "gamma")

    leader = max(rows, key=lambda row: row["selection_f1"])
    anchor = settings[leader["name"]]
    for name, drop in (("No HOG", {"hog"}), ("No local words", {"local"}),
                       ("Color texture and edges", {"hog", "local"})):
        kept = [(g, w) for g, w in zip(anchor.groups, anchor.weights, strict=True) if g not in drop]
        groups, weights = zip(*kept)
        run(replace(anchor, groups=groups, weights=weights),
            [(name, leader["c"], "mixed")], "features")

    promising = best[0]
    geometry = settings[promising["name"]]
    # Retain all original rows and change only their weights in the classifier fit.
    regularization_index = C_VALUES.index(promising["c"])
    weight_cs = C_VALUES[max(0, regularization_index - 1):regularization_index + 1]
    run(geometry, [(f"{name} C={c:g}", c, mode) for c in weight_cs for name, mode in (
        ("Source balanced", "balanced"), ("Source and class balanced", "balanced_classes"),
    )], "field weights")
    run(compact, [(COMPACT_NAME, compact.c, "balanced")], "benchmark", features(dev, compact))

    row, bundle = best
    # Promotion requires a gain on the declared score and a smaller gap for each source.
    promote = row["selection_f1"] > original["selection_f1"] and all(
        row[f"{source}_gap"] < original[f"{source}_gap"] for source in SOURCES
    )
    joblib.dump(bundle, MODEL, compress=3)
    pd.DataFrame([{**row, "promote_on_development": promote}]).to_csv(
        OUTPUT / "selected.csv", index=False
    )
    output = pd.concat(predictions, ignore_index=True)
    output.to_csv(OUTPUT / "development_predictions.csv", index=False)
    export_metrics(output, "development", OUTPUT)
    names = (BASELINE_NAME, row["name"], COMPACT_NAME)
    confusion_plot(output, names, "validation", SOURCES, "validation_confusions.png")
    confusion_plot(output, names, "cv", SOURCES, "cv_confusions.png")
    print(f"Selected: {row['name']} | promote on development: {promote}", flush=True)


def final_test():
    rows = pd.read_csv(HOLDOUT).assign(source="fresh field")
    selected = pd.read_csv(OUTPUT / "selected.csv").iloc[0]
    models = [(CURRENT, BASELINE_NAME), (MODEL, selected["name"]), (COMPACT, COMPACT_NAME)]
    predictions = []
    for path, name in models:
        bundle = joblib.load(path)
        experiment = Experiment(**bundle["experiment"])
        x = combine(parts_for(rows, experiment), experiment.groups)
        frame = rows[["path", "source", "label"]].assign(
            candidate=name, split="fresh test", predicted=predict(bundle, x)
        )
        predictions.append(frame)
        print(name, score(frame), flush=True)
    output = pd.concat(predictions, ignore_index=True)
    output.to_csv(OUTPUT / "fresh_predictions.csv", index=False)
    export_metrics(output, "fresh", OUTPUT)
    confusion_plot(output, [name for _, name in models], "fresh test", ("fresh field",),
                   "fresh_confusions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-test", action="store_true", help="Evaluate the fixed three models.")
    options = parser.parse_args()
    with threadpool_limits(limits=4):
        (final_test if options.fresh_test else development)()
