"""Evaluate the saved models and export tables, confusion matrices and figures."""

import argparse
from dataclasses import replace

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from dataset_preparation import CLASSES, FIGURES, RESULTS, ROOT, SEED
from feature_extraction import cached_features
from train_models import FILES, MODELS, Experiment, accuracy_and_f1, combine, fit_model, predict


def feature_matrix(rows, config, augment=None):
    mode = config["mode"]
    parts = cached_features(rows, mode, augment)
    if "local" in config["groups"]:
        parts.update(cached_features(rows, f"local-{mode}", augment))
    return combine(parts, config["groups"])


def test_predictions(rows, finalists, evaluation):
    frames = []
    for candidate in finalists.itertuples(index=False):
        bundle = joblib.load(MODELS / FILES[candidate.model])
        if bundle["experiment"]["name"] != candidate.name:
            raise ValueError("Saved model and CV selection disagree.")
        x = feature_matrix(rows, bundle["experiment"])
        frames.append(rows[["path", "label"]].assign(
            candidate=candidate.name, evaluation=evaluation, predicted=predict(bundle, x)
        ))
    return pd.concat(frames, ignore_index=True)


def metric_tables(predictions):
    metric_rows, matrix_rows = [], []
    for (name, split), rows in predictions.groupby(["candidate", "evaluation"]):
        report = classification_report(rows.label, rows.predicted, labels=CLASSES,
                                       output_dict=True, zero_division=0)
        for label in (*CLASSES, "macro avg"):
            metric_rows.append({"candidate": name, "evaluation": split, "label": label,
                                **report[label], "accuracy": report.get("accuracy",
                                                                     (rows.label == rows.predicted).mean())})
        counts = confusion_matrix(rows.label, rows.predicted, labels=CLASSES)
        for i, actual in enumerate(CLASSES):
            for j, prediction in enumerate(CLASSES):
                count = counts[i, j]
                matrix_rows.append({"candidate": name, "evaluation": split, "actual": actual,
                                    "predicted": prediction, "count": count,
                                    "row_percent": 100 * count / max(counts[i].sum(), 1)})
    pd.DataFrame(metric_rows).to_csv(RESULTS / "per_class_metrics.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(RESULTS / "confusion_matrices.csv", index=False)


def confusion_figures(predictions, finalists):
    labels = [label.title() for label in CLASSES]
    for candidate in finalists.itertuples(index=False):
        filtered = predictions[predictions.candidate == candidate.name]
        for split in ("cv", "validation", "internal_test", "external_test"):
            rows = filtered[filtered.evaluation == split]
            if rows.empty:
                continue
            counts = confusion_matrix(rows.label, rows.predicted, labels=CLASSES)
            percent = counts * 100 / np.maximum(counts.sum(axis=1, keepdims=True), 1)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.65), layout="constrained")
            for ax, matrix, title, fmt in zip(axes, (counts, percent), ("Count", "% of true class"),
                                               ("d", ".1f"), strict=True):
                ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
                    ax=ax, cmap="Greens", values_format=fmt, colorbar=False, xticks_rotation=20
                )
                ax.set_title(title)
            fig.suptitle(f"{candidate.model} · {split.replace('_', ' ').title()}")
            fig.savefig(FIGURES / f"{FILES[candidate.model].split('.')[0]}_{split}.png", dpi=155)
            plt.close(fig)


def comparison_table(selected, predicted):
    rows = []
    for model in selected.itertuples(index=False):
        tests = {}
        model_predictions = predicted[predicted.candidate == model.name]
        for split in ("internal_test", "external_test"):
            part = model_predictions[model_predictions.evaluation == split]
            tests[f"{split}_accuracy"], tests[f"{split}_macro_f1"] = (
                accuracy_and_f1(part.label, part.predicted) if len(part) else (np.nan, np.nan)
            )
        rows.append({
            "model": model.model, "candidate": model.name, "train_accuracy": model.train_accuracy,
            "cv_accuracy": model.cv_accuracy, "cv_accuracy_std": model.cv_accuracy_std,
            "cv_macro_f1": model.cv_macro_f1, "validation_accuracy": model.validation_accuracy,
            "validation_macro_f1": model.validation_macro_f1,
            "train_validation_gap": model.train_accuracy - model.validation_accuracy,
            **tests,
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "model_comparison.csv", index=False)
    columns = ["train_accuracy", "cv_accuracy", "validation_accuracy", "internal_test_accuracy",
               "external_test_accuracy"]
    widths = ["Train", "CV", "Validation", "Internal test", "External test"]
    chart = metrics.set_index("model")[columns].copy()
    chart.columns = widths
    chart = chart.dropna(axis=1, how="all")
    ax = chart.mul(100).plot.bar(figsize=(10, 5), rot=0, width=.8,
                               color=["#b4c8a4", "#70917d", "#466858", "#28433e", "#cc7359"])
    ax.set(ylim=(0, 108), ylabel="Accuracy (%)", xlabel="", title="Flowers model comparison")
    ax.axhline(80, linestyle="--", color="#71665b", linewidth=1)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES / "model_comparison.png", dpi=155)
    plt.close(ax.figure)
    print(metrics.to_string(index=False), flush=True)


def learning_curve(manifest, config, dimensions):
    train = manifest[manifest.split == "train"].reset_index(drop=True)
    val = manifest[manifest.split == "validation"]
    dev = manifest[manifest.split.isin(["train", "validation"])]
    features = feature_matrix(dev, config)
    x_train = features[dev.split == "train"]
    x_val = features[dev.split == "validation"]
    copies = [feature_matrix(train, config, method) for method in config["augment"]]
    folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    chunks = [hold for _, hold in folds.split(train, train.label, train.group)]
    experiment = Experiment(**config)
    rows = []
    for count in (1, 2, 3, 4):
        indices = np.sort(np.concatenate(chunks[:count]))
        y = train.label.iloc[indices].to_numpy(dtype=str)
        x = np.vstack([x_train[indices], *[a[indices] for a in copies]])
        # Nyström uses at most one landmark per available fitting row.
        subset = replace(experiment, components=min(experiment.components, len(x)))
        bundle = fit_model(subset, dimensions, x, np.tile(y, 1 + len(copies)))
        train_accuracy, _ = accuracy_and_f1(y, predict(bundle, x_train[indices]))
        val_accuracy, _ = accuracy_and_f1(val.label, predict(bundle, x_val))
        rows.append({"original_training_images": len(indices),
                     "landmarks": subset.components if subset.kernel else 0,
                     "train_accuracy": train_accuracy,
                     "validation_accuracy": val_accuracy})
    curve = pd.DataFrame(rows)
    curve.to_csv(RESULTS / "learning_curve.csv", index=False)
    ax = curve.set_index("original_training_images").mul(100).plot(
        marker="o", figsize=(7, 4), color=["#70917d", "#cc7359"]
    )
    ax.set(xlabel="Original training images", ylabel="Accuracy (%)",
           title="Selected model learning curve")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES / "learning_curve.png", dpi=155)
    plt.close(ax.figure)


def evaluate(external, curve):
    selected = pd.read_csv(RESULTS / "selected_models.csv")
    manifest = pd.read_csv(RESULTS / "manifest.csv")
    FIGURES.mkdir(exist_ok=True)
    test = manifest[manifest.split == "test"]
    predictions = pd.read_csv(RESULTS / "development_predictions.csv")
    internal = test_predictions(test, selected, "internal_test")
    internal.to_csv(RESULTS / "internal_test_predictions.csv", index=False)
    frames = [predictions, internal]
    if external:
        rows = pd.read_csv(RESULTS / "external_manifest.csv")
        result = test_predictions(rows, selected.head(1), "external_test")
        result.to_csv(RESULTS / "external_test_predictions.csv", index=False)
        frames.append(result)
    predictions = pd.concat(frames, ignore_index=True)
    metric_tables(predictions)
    confusion_figures(predictions, selected)
    comparison_table(selected, predictions)
    if curve:
        winner = joblib.load(MODELS / FILES[selected.iloc[0].model])
        learning_curve(manifest, winner["experiment"], winner["dimensions"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", action="store_true",
                        help="Score the selected winner on the prepared independent holdout.")
    parser.add_argument("--learning-curve", action="store_true",
                        help="Refit the selected model on four group-preserving subsets.")
    options = parser.parse_args()
    with threadpool_limits(limits=4):
        evaluate(options.external, options.learning_curve)
