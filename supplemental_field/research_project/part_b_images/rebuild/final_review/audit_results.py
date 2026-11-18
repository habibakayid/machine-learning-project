"""Consolidate saved results and verify deployment bundles on development photos only."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits

from audit_data import CLASSES, EXTERNAL, GENERAL, OUT, RAW, RESULTS, ROOT
from generalization_data import FIELD
from generalization_study import parts_for
from train_models import Experiment, FILES, MODELS, combine, predict_image
import matplotlib.pyplot as plt

ARCHIVE = GENERAL / "models"
REBUILD = OUT.parent
MODELS_BY_ID = {
    "lr_original": MODELS / FILES["Logistic Regression"],
    "lr_mixed": ARCHIVE / FILES["Logistic Regression"],
    "lr_c3": REBUILD / "logistic_study" / "regularized_logistic.joblib",
    "lr_compact": ARCHIVE / "compact_logistic.joblib",
    "knn_rebuilt": MODELS / FILES["KNN Classifier"],
    "knn_archived": ARCHIVE / FILES["KNN Classifier"],
    "knn_final": REBUILD / "knn_study" / FILES["KNN Classifier"],
    "km_rebuilt": MODELS / FILES["K-Means"],
    "km_archived": ARCHIVE / FILES["K-Means"],
    "km_final": REBUILD / "kmeans_study" / "promoted_kmeans.joblib",
}
FINAL = {"lr_mixed": "Nyström Logistic Regression + field training",
         "knn_final": "Manhattan KNN with Nyström and LDA",
         "km_final": "15-cluster K-Means with local visual words"}
KAGGLE_ONLY = {"lr_original", "knn_rebuilt", "km_rebuilt"}
# Integer counts transcribed from the three original Test Confusion Matrix PNGs.
LEGACY = {
    "Logistic Regression": ("legacy_lr", [[49,32,14,9,11], [27,97,8,16,10], [8,11,68,12,18],
                                          [8,23,3,65,11], [8,20,16,11,92]]),
    "KNN Classifier": ("legacy_knn", [[48,21,13,9,24], [17,73,15,16,37], [13,16,27,4,57],
                                     [9,8,9,55,29], [10,16,15,7,99]]),
    "K-Means": ("legacy_km", [[0,38,0,28,49], [0,67,0,23,68], [0,38,0,24,55],
                             [0,33,0,21,56], [0,40,0,18,89]]),
}
INITIAL = {"Global + local quadratic": "lr_original", "KNN with local descriptors": "knn_rebuilt",
           "K-Means all, one-to-one": "km_rebuilt"}
GEN_MODELS = {
    "Existing Nystrom": "lr_original", "Nystrom mixed": "lr_mixed",
    "Foreground crop balanced C=1": "lr_compact", "KNN compact mixed": "knn_archived",
    "K-Means compact mixed": "km_archived",
}
SAVED_CV = [
    ("results", {k: v for k, v in INITIAL.items() if v != "lr_original"}),
    ("generalization", {"Existing Nystrom": "lr_original", "Nystrom mixed": "lr_mixed",
                        "Foreground crop balanced C=1": "lr_compact"}),
    ("logistic_study", {"Source and class balanced C=3": "lr_c3"}),
    ("knn_study", {"Archived KNN": "knn_archived", "Selected KNN": "knn_final"}),
    ("kmeans_study", {"Archived K-Means": "km_archived", "Selected K-Means": "km_final"}),
]


def source_csv(folder, filename):
    raw = pd.read_csv(REBUILD / folder / filename)
    return raw.rename(columns={"evaluation": "split"}).assign(
        source=raw["source"] if "source" in raw else "kaggle")


def pooled_split(values):
    return values.str.replace(r"^cv_\d+$", "cv", regex=True)


def matrix_cells(matrix):
    for i, actual in enumerate(CLASSES):
        for j, predicted in enumerate(CLASSES):
            value = int(matrix[i, j])
            yield dict(actual=actual, predicted=predicted, count=value,
                       row_percent=100 * value / max(1, matrix[i].sum()))


def historical_matrix_checks():
    sources = [
        ("results", "confusion_matrices.csv", ("development_predictions.csv",
          "internal_test_predictions.csv", "external_test_predictions.csv")),
        ("generalization", "development_confusions.csv", ("development_predictions.csv",)),
        ("generalization", "fresh_test_confusions.csv", ("fresh_test_predictions.csv",)),
        ("knn_study", "development_confusions.csv", ("development_predictions.csv",)),
        ("knn_study", "fresh_confusions.csv", ("fresh_predictions.csv",)),
        ("logistic_study", "development_confusions.csv", ("development_predictions.csv",)),
        ("logistic_study", "fresh_confusions.csv", ("fresh_predictions.csv",)),
        ("kmeans_study", "development_confusions.csv", ("development_predictions.csv",)),
    ]
    keys = ["candidate", "source", "split"]
    checks = []
    for folder, name, files in sources:
        saved = source_csv(folder, name)
        assert not saved.duplicated([*keys, "actual", "predicted"]).any()
        predictions = pd.concat([source_csv(folder, file) for file in files], ignore_index=True)
        predictions["split"] = pooled_split(predictions.split)
        assert not predictions.duplicated([*keys, "path"]).any()
        actual = []
        for key, rows in predictions.groupby(keys):
            assert rows.label.isin(CLASSES).all() and rows.predicted.isin(CLASSES).all()
            matrix = confusion_matrix(rows.label, rows.predicted, labels=CLASSES)
            assert matrix.sum() == len(rows) and matrix.trace() == rows.label.eq(rows.predicted).sum()
            actual.extend({**dict(zip(keys, key)), **cell} for cell in matrix_cells(matrix))
        fields = [*keys, "actual", "predicted"]
        expected = pd.DataFrame(actual).set_index(fields).sort_index()
        reference = saved.set_index(fields).sort_index()
        pd.testing.assert_index_equal(reference.index, expected.index)
        assert np.array_equal(reference["count"], expected["count"])
        assert np.allclose(reference.row_percent, expected.row_percent)
        checks.append({"file": f"{folder}/{name}", "matrices": len(expected) // 25,
                       "cells": len(expected), "result": "matches saved predictions"})
    pd.DataFrame(checks).to_csv(OUT / "CONFUSION_CHECKS.csv", index=False)
    print(f"Verified {sum(row['matrices'] for row in checks)} saved prediction-backed matrices.",
          flush=True)


def fixed_development():
    rows = pd.read_csv(GENERAL / "development_split.csv")
    manifests = [pd.read_csv(RESULTS / "manifest.csv"), pd.read_csv(FIELD)]
    lookup = pd.concat(manifests).set_index("path")
    assert rows.path.is_unique and rows.split.isin(["train", "validation"]).all()
    assert rows.path.isin(lookup.index).all()
    assert rows.label.tolist() == lookup.loc[rows.path, "label"].tolist()
    rows["sha256"] = rows.path.map(lookup.sha256)
    return rows


def check_membership(predictions, reference):
    assert predictions.path.is_unique and set(predictions.path) == set(reference.path)
    actual = reference.set_index("path").loc[predictions.path, "label"]
    assert actual.tolist() == predictions.label.tolist()


def development_predictions(dev):
    frames, manifest, fixtures = [], [], []
    cache = {}
    for model, path in MODELS_BY_ID.items():
        bundle = joblib.load(path)
        experiment = Experiment(**bundle["experiment"])
        pipe = bundle["pipeline"]
        if bundle["mapping"] is None:
            assert tuple(pipe.classes_) == CLASSES
        else:
            assert set(bundle["mapping"]) == set(CLASSES)
            assert len(bundle["mapping"]) == pipe.named_steps["estimator"].n_clusters
        assert sum(bundle["dimensions"]) == pipe.n_features_in_
        if model == "km_final":
            cluster_counts = pd.read_csv(REBUILD / "kmeans_study" / "cluster_train_counts.csv")
            assert list(cluster_counts.cluster) == list(range(15))
            assert cluster_counts[list(CLASSES)].to_numpy().sum() == 3467
            assert cluster_counts[list(CLASSES)].idxmax(axis=1).tolist() == bundle["mapping"].tolist()
            assert bundle["promotion_on_development"]
        manifest.append(dict(
            model=model, final=model in FINAL, relative_path=path.relative_to(ROOT).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mode=experiment.mode,
            groups=list(experiment.groups), dimensions=bundle["dimensions"],
            input_features=pipe.n_features_in_,
            estimator_features=pipe.named_steps["estimator"].n_features_in_,
            steps=list(pipe.named_steps), estimator=type(pipe.steps[-1][1]).__name__,
            estimator_parameters=pipe.steps[-1][1].get_params(deep=False),
        ))
        for source, rows in dev.groupby("source", sort=False):
            key = (source, experiment.mode, "local" in experiment.groups)
            if key not in cache:
                cache[key] = parts_for(rows, experiment)
            x = combine(cache[key], experiment.groups)
            assigned = pipe.predict(x)
            predicted = assigned if bundle["mapping"] is None else bundle["mapping"][assigned]
            frame = rows[["path", "source", "label", "split"]].assign(
                model=model, predicted=predicted, provenance="new development inference",
                fold=pd.NA,
            )
            if model in KAGGLE_ONLY and source == "field":
                frame = frame[frame.split.eq("validation")]
            frames.append(frame)
            if model in FINAL and source == "kaggle":
                sample = rows[rows.split.eq("validation")].groupby("label", sort=True).head(1)
                for item in sample.itertuples():
                    i = rows.index.get_loc(item.Index)
                    with Image.open(ROOT / item.path) as image:
                        label, probs, cluster = predict_image(bundle, image)
                    assert label == predicted[i]
                    if probs is not None:
                        expected = pipe.predict_proba(x[i:i+1])[0]
                        assert tuple(probs) == CLASSES and np.allclose(list(probs.values()), expected, atol=1e-5)
                    else:
                        assert cluster == assigned[i]
                    fixtures.append({"model": model, "actual": item.label, "path": item.path,
                                     "predicted": label, "probabilities": probs,
                                     "cluster_display": None if cluster is None else cluster + 1})
    for folder, mapping in SAVED_CV:
        saved = source_csv(folder, "development_predictions.csv")
        saved = saved[saved.candidate.isin(mapping) & saved.split.str.startswith("cv")]
        for (_, source), part in saved.groupby(["candidate", "source"]):
            ref = dev[dev.source.eq(source) & dev.split.eq("train")]
            check_membership(part, ref)
            archived_fold = part.split.str.extract(r"^cv_(\d+)$")[0].astype(float)
            stored_fold = ref.set_index("path").loc[part.path, "fold"].to_numpy()
            if archived_fold.notna().any():
                assert np.array_equal(archived_fold.to_numpy(), stored_fold)
            frames.append(part[["path", "source", "label", "predicted"]].assign(
                model=part.candidate.map(mapping), split="cv", fold=stored_fold,
                provenance="saved out-of-fold predictions",
            ))
    current = pd.concat(frames, ignore_index=True)
    for folder, mapping in SAVED_CV:
        saved = source_csv(folder, "development_predictions.csv")
        saved = saved[saved.candidate.isin(mapping) & saved.split.isin(["train", "validation"])]
        saved = saved.assign(model=saved.candidate.map(mapping))
        keys = ["model", "source", "split", "path"]
        for key, reference in saved.groupby(keys[:3]):
            actual = current[(current.model == key[0]) & (current.source == key[1])
                             & (current.split == key[2])]
            check_membership(actual, reference)
            assert actual.predicted.tolist() == reference.set_index("path").loc[actual.path, "predicted"].tolist()
    (OUT / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "DEMO_FIXTURES.json").write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
    return current


def test_predictions():
    tests = [
        ("kaggle_internal_645", "results", "internal_test_predictions.csv", INITIAL),
        ("field_2025_234", "results", "external_test_predictions.csv", INITIAL),
        ("field_2026_150", "generalization", "fresh_test_predictions.csv", GEN_MODELS),
        ("field_2020_2021_191", "knn_study", "fresh_predictions.csv",
         {"Archived KNN": "knn_archived", "Selected KNN": "knn_final",
          "Frozen Logistic Regression": "lr_mixed"}),
        ("field_2016_2019_156", "logistic_study", "fresh_predictions.csv",
         {"Current Nystrom": "lr_mixed", "Source and class balanced C=3": "lr_c3",
          "Compact benchmark": "lr_compact"}),
    ]
    frames = []
    for name, folder, file, mapping in tests:
        raw = source_csv(folder, file)
        ref = pd.read_csv(EXTERNAL[name]) if name in EXTERNAL else pd.read_csv(
            RESULTS / "manifest.csv").query("split == 'test'")
        assert raw.candidate.isin(mapping).all()
        for _, rows in raw.groupby("candidate"):
            check_membership(rows, ref)
        frames.append(raw.assign(model=raw.candidate.map(mapping), split=name, fold=pd.NA,
                                 source="field" if name in EXTERNAL else "kaggle",
                                 provenance="saved test predictions"))
    return pd.concat(frames, ignore_index=True)


def scores(rows):
    report = classification_report(rows.label, rows.predicted, labels=CLASSES,
                                   zero_division=0, output_dict=True)
    return {"accuracy": rows.label.eq(rows.predicted).mean(),
            **{f"macro_{metric}": report["macro avg"][key] for metric, key in
               (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score"))}}, report


def tables(rows, fields):
    summary, classes, counts = [], [], []
    for key, subset in rows.groupby(fields):
        key_fields = dict(zip(fields, key))
        assert subset.path.is_unique
        pooled, report = scores(subset)
        values = pooled
        if key_fields["split"] == "cv":
            values = pd.DataFrame([scores(fold)[0] for _, fold in subset.groupby("fold")]).mean().to_dict()
        summary.append({**key_fields, **values, "correct": int(subset.label.eq(subset.predicted).sum()),
                        "n": len(subset), "pooled_accuracy": pooled["accuracy"]})
        classes.extend({**key_fields, "label": label, **report[label]} for label in CLASSES)
        matrix = confusion_matrix(subset.label, subset.predicted, labels=CLASSES)
        counts.extend({**key_fields, **cell} for cell in matrix_cells(matrix))
    return pd.DataFrame(summary), pd.DataFrame(classes), pd.DataFrame(counts)


def legacy_tables():
    folder = ROOT / "part_b_images" / "results"
    saved = pd.read_csv(folder / "final_test_summary.csv").set_index("model")
    per_class = pd.read_csv(folder / "final_test_per_class.csv")
    manifest = pd.read_csv(ROOT / "part_b_images" / "data" / "processed" / "splits" / "test.csv")
    support = manifest.label.value_counts().loc[list(CLASSES)].to_numpy()
    mixture = pd.read_csv(folder / "kmeans_test_clusters.csv").set_index("cluster")
    cluster_map = json.loads((ROOT / "models" / "part_b_kmeans_combined_mapping.json").read_text())
    mapped = np.zeros((5, 5), dtype=int)
    for cluster, row in mixture.iterrows():
        mapped[:, CLASSES.index(cluster_map[str(cluster)])] += row[list(CLASSES)].to_numpy(dtype=int)
    summaries, classes, counts = [], [], []
    for name, (model, matrix) in LEGACY.items():
        matrix = np.array(matrix)
        assert np.array_equal(matrix.sum(axis=1), support)
        if model == "legacy_km":
            assert np.array_equal(matrix, mapped)
        pairs = list(matrix_cells(matrix))
        expanded = pd.DataFrame(pairs).rename(columns={"actual": "label"})
        expanded = expanded.loc[expanded.index.repeat(expanded["count"])]
        metrics, report = scores(expanded)
        assert all(np.isclose(metrics[metric], saved.at[name, metric]) for metric in metrics)
        key = dict(model=model, source="kaggle", split="legacy_kaggle_test_647",
                   provenance="historical PNG counts checked against saved metrics")
        summaries.append({**key, **metrics, "correct": matrix.trace(), "n": matrix.sum(),
                          "pooled_accuracy": metrics["accuracy"]})
        for label in CLASSES:
            verify = per_class[(per_class.model == name) & (per_class["class"] == label)].iloc[0]
            for field, column in (("precision", "precision"), ("recall", "recall"),
                                  ("f1-score", "f1_score"), ("support", "support")):
                assert np.isclose(report[label][field], verify[column])
            classes.append({**key, "label": label, **report[label]})
        counts.extend({**key, **item} for item in pairs)
    return pd.DataFrame(summaries), pd.DataFrame(classes), pd.DataFrame(counts)


def figures(counts):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    errors = []
    for model, title in FINAL.items():
        for source in ("kaggle", "field"):
            subset = counts[counts.model.eq(model) & counts.source.eq(source)]
            fig, axes = plt.subplots(3, 2, figsize=(9, 12), layout="constrained")
            for i, split in enumerate(("train", "cv", "validation")):
                table = subset[subset.split.eq(split)]
                for j, (key, units, fmt) in enumerate((
                    ("count", "counts", "d"), ("row_percent", "% of true class", ".1f"),
                )):
                    values = table.pivot(index="actual", columns="predicted", values=key).loc[CLASSES, CLASSES]
                    options = {"vmin": 0, "vmax": 100} if j == 1 else {}
                    ax = axes[i, j]
                    ConfusionMatrixDisplay(values.to_numpy(), display_labels=[c.title() for c in CLASSES]).plot(
                        ax=ax, cmap="Greens", colorbar=False, values_format=fmt, xticks_rotation=25, im_kw=options)
                    name = "Grouped CV, pooled" if split == "cv" else split.title()
                    ax.set_title(f"{name} · {units}")
                if split == "validation":
                    for item in table[table.actual.ne(table.predicted)].nlargest(3, "count").to_dict("records"):
                        errors.append(item)
            fig.suptitle(f"{title} · {source.title()}")
            fig.savefig(folder / f"{model}_{source}.png", dpi=130)
            plt.close(fig)
    pd.DataFrame(errors).to_csv(OUT / "TOP_ERRORS.csv", index=False)
    original = source_csv("logistic_study", "development_confusions.csv")
    study_names = ("Current Nystrom", "Source and class balanced C=3", "Compact benchmark")
    for split in ("cv", "validation"):
        figure, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
        rows = original[original.split.eq(split)]
        pairs = [(source, name) for source in ("kaggle", "field") for name in study_names]
        for ax, (source, name) in zip(axes.flat, pairs, strict=True):
            part = rows[rows.source.eq(source) & rows.candidate.eq(name)]
            matrix = part.pivot(index="actual", columns="predicted", values="row_percent").loc[CLASSES, CLASSES]
            ConfusionMatrixDisplay(matrix.to_numpy(), display_labels=[c.title() for c in CLASSES]).plot(
                ax=ax, cmap="Greens", values_format=".0f", colorbar=False, xticks_rotation=25,
                im_kw={"vmin": 0, "vmax": 100})
            ax.set_title(f"{source.title()} · {name}", fontsize=10)
        title = "grouped CV, pooled" if split == "cv" else "validation"
        figure.suptitle(f"Logistic Regression study · {title} · % of true class")
        figure.savefig(folder / f"logistic_study_{split}_labeled.png", dpi=130)
        plt.close(figure)


def main():
    historical_matrix_checks()
    dev_rows = fixed_development()
    development = development_predictions(dev_rows)
    keys = ["model", "source", "split", "provenance"]
    dev, per_class, dev_counts = tables(development, keys)
    counts_by_split = dev.pivot(index=["model", "source"], columns="split", values="accuracy")
    gap = (counts_by_split.train - counts_by_split.validation).rename("train_validation_gap")
    dev = dev.join(gap, on=["model", "source"])
    dev.to_csv(OUT / "FINAL_METRICS.csv", index=False)
    per_class.to_csv(OUT / "FINAL_PER_CLASS.csv", index=False)
    dev_counts.to_csv(OUT / "FINAL_CONFUSIONS.csv", index=False)
    saved_tables = tables(test_predictions(), keys)
    tests, per_class, counts = (pd.concat([saved, archive], ignore_index=True)
                               for saved, archive in zip(saved_tables, legacy_tables(), strict=True))
    tests.to_csv(OUT / "TEST_METRICS.csv", index=False)
    per_class.to_csv(OUT / "TEST_PER_CLASS.csv", index=False)
    counts.to_csv(OUT / "TEST_CONFUSIONS.csv", index=False)
    figures(dev_counts)
    print(f"Consolidated {len(dev)} development and {len(tests)} historical test rows.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
