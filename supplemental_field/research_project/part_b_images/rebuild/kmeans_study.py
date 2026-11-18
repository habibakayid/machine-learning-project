"""Tune K-Means on the saved grouped development folds."""

from dataclasses import asdict, dataclass, replace
from itertools import product
import json
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (ConfusionMatrixDisplay, adjusted_rand_score, confusion_matrix,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from dataset_preparation import CLASSES, RESULTS, SEED
from generalization_data import FIELD
from generalization_study import (OUTPUT as GENERAL, STUDY_MODELS, export_metrics, parts_for)
from train_models import ALL, FILES, Experiment, accuracy_and_f1, combine, make_pipeline

OUTPUT = RESULTS.parent / "kmeans_study"
ORIGINAL = STUDY_MODELS / FILES["K-Means"]
MODEL = OUTPUT / FILES["K-Means"]
PROMOTED = OUTPUT / "promoted_kmeans.joblib"
SOURCES = ("kaggle", "field")
LABELS = np.asarray(CLASSES)
BASELINE = "Archived compact"


@dataclass(frozen=True)
class Settings:
    features: Experiment
    clusters: int = 5
    starts: int = 10
    balanced_sources: bool = False


def development_rows():
    dev = pd.read_csv(GENERAL / "development_split.csv")
    lookup = pd.concat([pd.read_csv(RESULTS / "manifest.csv"),
                        pd.read_csv(FIELD)]).set_index("path")
    dev["sha256"] = dev.path.map(lookup.sha256)
    assert dev.path.is_unique and dev.sha256.notna().all()
    assert set(dev.split) == {"train", "validation"}
    assert dev.groupby("group").split.nunique().max() == 1
    assert dev.groupby("group").fold.nunique().max() == 1
    assert dev.groupby(["source", "split"]).label.nunique().eq(len(CLASSES)).all()
    return dev


def cluster_labels(cluster, labels, n_clusters):
    """Assign names after fitting, using only the fitting partition's labels."""
    counts = np.zeros((n_clusters, len(CLASSES)), dtype=int)
    np.add.at(counts, (cluster, np.searchsorted(LABELS, labels)), 1)
    if n_clusters == len(CLASSES):
        ids, names = linear_sum_assignment(counts, maximize=True)
        assert np.array_equal(ids, np.arange(n_clusters))
    else:
        names = counts.argmax(axis=1)
    return LABELS[names]


def source_weights(rows):
    counts = rows.source.value_counts()
    return rows.source.map(len(rows) / (len(counts) * counts)).to_numpy()


def silhouette(coords, clusters):
    # Limit the quadratic distance matrix on Kaggle; smaller partitions use all rows.
    if len(coords) > 800:
        indices = np.random.default_rng(SEED).choice(len(coords), size=800, replace=False)
        coords, clusters = coords[indices], clusters[indices]
    if not 1 < len(np.unique(clusters)) < len(coords):
        return np.nan
    return silhouette_score(coords, clusters)


def metrics(part, coords):
    accuracy, f1 = accuracy_and_f1(part.label, part.predicted)
    return {
        "accuracy": accuracy, "macro_f1": f1,
        "silhouette": silhouette(coords, part.cluster.to_numpy()),
        "ari": adjusted_rand_score(part.label, part.cluster),
        "nmi": normalized_mutual_info_score(part.label, part.cluster),
        "classes_emitted": part.predicted.nunique(), "n": len(part),
    }


def compare(dev, raw, settings, folds):
    """Share each fold's unsupervised preprocessing across cluster settings."""
    exp = settings[0].features
    dimensions = [raw[name].shape[1] for name in exp.groups]
    x = combine(raw, exp.groups)
    train = np.flatnonzero(dev.split == "train")
    val = np.flatnonzero(dev.split == "validation")
    y = dev.label.to_numpy(dtype=str)
    ids = {s: [] for s in settings}
    bundles = {}
    rows = {s: [] for s in settings}

    def evaluate(spec, split, indices, coords, clusters, mapping):
        frame = dev.iloc[indices][["path", "source", "label"]].assign(
            split=split, predicted=mapping[clusters], cluster=clusters
        )
        ids[spec].append(frame)
        for source in SOURCES:
            mask = frame.source.eq(source).to_numpy()
            rows[spec].append({"source": source, "split": split, **metrics(frame[mask], coords[mask])})

    for fold, (fit, held) in enumerate([*folds, (train, val)], 1):
        final = fold == len(folds) + 1
        transform = make_pipeline(exp, dimensions)[:-1]
        # No flower labels enter the fitted scaler, PCA, codebook, or centroids.
        fit_x = transform.fit_transform(x[fit])
        held_x = transform.transform(x[held])
        for spec in settings:
            estimator = KMeans(n_clusters=spec.clusters, init="k-means++",
                               n_init=spec.starts, random_state=SEED)
            weight = source_weights(dev.iloc[fit]) if spec.balanced_sources else None
            estimator.fit(fit_x, sample_weight=weight)
            mapping = cluster_labels(estimator.labels_, y[fit], spec.clusters)
            if final:
                bundles[spec] = {
                    "pipeline": Pipeline([*transform.steps, ("estimator", estimator)]),
                    "experiment": asdict(replace(exp, n_init=spec.starts,
                                                one_to_one=spec.clusters == len(CLASSES))),
                    "dimensions": dimensions, "mapping": mapping,
                }
                evaluate(spec, "train", fit, fit_x, estimator.labels_, mapping)
            split = "validation" if final else f"cv_{fold}"
            evaluate(spec, split, held, held_x, estimator.predict(held_x), mapping)

    for spec in settings:
        detailed = pd.DataFrame(rows[spec])
        summary = {}
        for source, split in product(SOURCES, ("train", "cv", "validation")):
            subset = detailed[(detailed.source == source) & detailed.split.str.startswith(split)]
            for metric in ("accuracy", "macro_f1", "silhouette", "ari", "nmi"):
                summary[f"{source}_{split}_{metric}"] = subset[metric].mean()
        bundle = bundles[spec]
        summary["classes_in_mapping"] = len(set(bundle["mapping"]))
        summary["selection_f1"] = np.mean([summary[f"{s}_cv_macro_f1"] for s in SOURCES])
        summary["validation_f1"] = np.mean([summary[f"{s}_validation_macro_f1"] for s in SOURCES])
        yield spec, summary, bundle, pd.concat(ids[spec], ignore_index=True), detailed


def representations(baseline):
    crop = replace(baseline, mode="crop", pca=20, scale="l2")
    compact = replace(baseline, pca=20)
    return [
        replace(compact, name="Compact pad standardized"),
        replace(compact, name="Compact crop normalized", mode="compact-crop", scale="l2"),
        replace(compact, name="Foreground and global color", mode="compact-crop", scale="l2",
                groups=("global_color", "foreground_color"), weights=(1., 1.4)),
        replace(crop, name="Joint color", groups=("color",), weights=(1.,)),
        replace(crop, name="HOG", groups=("hog",), weights=(1.,)),
        replace(crop, name="LBP", groups=("lbp",), weights=(1.,)),
        replace(crop, name="Multiscale edges", groups=("edges",), weights=(1.,)),
        replace(crop, name="Local visual words", groups=("local",), weights=(1.,)),
        replace(crop, name="Color and texture", groups=("color", "lbp"), weights=(1.7, 1.)),
        replace(crop, name="Global handcrafted", groups=ALL, weights=(1.8, .6, .8, .8)),
        replace(crop, name="Global and local", groups=(*ALL, "local"),
                weights=(1.8, .55, .75, .75, 1.)),
        replace(crop, name="Color texture and local", groups=("color", "lbp", "local"),
                weights=(1.8, .9, 1.)),
    ]


def plot_confusions(rows, split):
    names = rows.candidate.unique()
    fig, axes = plt.subplots(2, len(names), squeeze=False, figsize=(4.5 * len(names), 8),
                             layout="constrained")
    for i, source in enumerate(SOURCES):
        for j, name in enumerate(names):
            subset = rows[(rows.source == source) & (rows.candidate == name)
                          & rows.split.str.startswith(split)]
            counts = confusion_matrix(subset.label, subset.predicted, labels=CLASSES)
            normalized = 100 * counts / np.maximum(1, counts.sum(axis=1, keepdims=True))
            ax = axes[i, j]
            ConfusionMatrixDisplay(normalized, display_labels=[c.title() for c in CLASSES]).plot(
                ax=ax, cmap="Greens", colorbar=False, values_format=".0f", xticks_rotation=25,
                im_kw={"vmin": 0, "vmax": 100},
            )
            ax.set_title(f"{source.title()} · {name}")
    fig.suptitle(f"{split.upper()} confusion matrices, percent of each true class")
    fig.savefig(OUTPUT / f"{split}_confusions.png", dpi=140)
    plt.close(fig)


def study():
    OUTPUT.mkdir(exist_ok=True)
    dev = development_rows()
    snapshot = joblib.load(ORIGINAL)
    baseline = replace(Experiment(**snapshot["experiment"]), name=BASELINE)
    train = np.flatnonzero(dev.split == "train")
    folds = [(train[dev.fold.iloc[train] != i], train[dev.fold.iloc[train] == i])
             for i in (1, 2, 3)]
    records, raw_cache, all_rows, all_details = {}, {}, [], []
    best = None

    def run(settings, stage):
        nonlocal best
        settings = [s for s in settings if s not in records]
        if not settings:
            return
        exp = settings[0].features
        key = exp.mode, "local" in exp.groups
        if key not in raw_cache:
            chunks = [parts_for(dev[dev.source == source], exp) for source in SOURCES]
            assert dev.source.tolist() == sorted(dev.source, key=SOURCES.index)
            raw_cache[key] = {name: np.concatenate([part[name] for part in chunks])
                              for name in chunks[0]}
        for spec, summary, bundle, frames, details in compare(dev, raw_cache[key], settings, folds):
            row = {
                "id": len(all_rows) + 1, "stage": stage, "representation": exp.name,
                "mode": exp.mode, "groups": exp.groups, "weights": exp.weights, "scale": exp.scale,
                "pca": exp.pca, "clusters": spec.clusters, "n_init": spec.starts,
                "balanced_sources": spec.balanced_sources, **summary,
            }
            records[spec] = row
            all_rows.append(row)
            all_details.append(details.assign(id=row["id"]))
            if stage == "baseline":
                x = combine(raw_cache[key], exp.groups)
                known = snapshot["mapping"][snapshot["pipeline"].predict(x)]
                validation = dev.split.eq("validation")
                assert np.array_equal(known[validation], frames[frames.split == "validation"].predicted)
                baseline_frames.append(frames.assign(candidate="Archived K-Means"))
            elif row["classes_in_mapping"] == len(CLASSES) and (
                best is None or row["selection_f1"] > best[0]["selection_f1"]
            ):
                best = row, bundle, frames.assign(candidate="Selected K-Means"), spec
            print(f"{exp.name} | PCA {exp.pca} | k={spec.clusters} | starts={spec.starts}"
                  f" | weighted={spec.balanced_sources} | CV F1 {row['selection_f1']:.2%}"
                  f" | {row['classes_in_mapping']} mapped classes", flush=True)
        pd.DataFrame(all_rows).to_csv(OUTPUT / "experiments.csv", index=False)

    baseline_frames = []
    old_spec = Settings(baseline, starts=snapshot["pipeline"].named_steps["estimator"].n_init)
    run([old_spec], "baseline")
    features = representations(baseline)
    for exp in features:
        run([Settings(exp, k) for k in (5, 15)], "features")

    def leading(experiments):
        return sorted(experiments, key=lambda exp: max(
            records[Settings(exp, k)]["selection_f1"] for k in (5, 15)
        ), reverse=True)

    shortlist = leading(features)[:3]
    for exp, size in product(shortlist, (20, 40, 80, 120, 0)):
        geometry = replace(exp, pca=size)
        run([Settings(geometry, k) for k in (5, 10, 15)], "PCA and clusters")

    multi = max((s for s in records if s != old_spec and len(s.features.groups) > 1),
                key=lambda s: records[s]["selection_f1"])
    exp = multi.features
    color_groups = {"color", "global_color", "foreground_color"}
    for focus in (color_groups, set(exp.groups) - color_groups):
        if not any(group in focus for group in exp.groups) or all(group in focus for group in exp.groups):
            continue
        weight = tuple(value * (1.7 if group in focus else 1)
                       for group, value in zip(exp.groups, exp.weights, strict=True))
        name = "more color" if focus == color_groups else "more shape or texture"
        run([replace(multi, features=replace(exp, name=f"{exp.name}, {name}", weights=weight))],
            "group weights")
    alt_scale = "l2" if exp.scale == "standard" else "standard"
    run([replace(multi, features=replace(exp, name=f"{exp.name}, {alt_scale}", scale=alt_scale))],
        "scaling")

    # Check whether a small amount of global information helps the leading local geometry.
    local = max((s for s in records if s.features.groups == ("local",)),
                key=lambda s: records[s]["selection_f1"])
    for groups, weights, name in (
        (("local", "color"), (1., .25), "Local with joint color"),
        (("local", "lbp", "edges"), (1., .2, .2), "Local with texture and edges"),
    ):
        run([replace(local, features=replace(local.features, name=name, groups=groups, weights=weights))],
            "local feature weights")

    finalists = sorted((s for s in records if s != old_spec and records[s]["classes_in_mapping"] == 5),
                       key=lambda s: records[s]["selection_f1"], reverse=True)[:2]
    for spec in finalists:
        run([replace(spec, starts=20), replace(spec, balanced_sources=True)],
            "initialization and source weights")

    row, bundle, predictions, _ = best
    previous = records[old_spec]
    promote = (row["selection_f1"] > previous["selection_f1"]
               and row["validation_f1"] > previous["validation_f1"])
    bundle["promotion_on_development"] = bool(promote)
    joblib.dump(bundle, MODEL, compress=3)
    if promote:
        pending = PROMOTED.with_suffix(".tmp")
        shutil.copyfile(MODEL, pending)
        pending.replace(PROMOTED)
    names = {str(i): str(label) for i, label in enumerate(bundle["mapping"])}
    (OUTPUT / "cluster_labels.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{**row, "promote_on_development": promote}]).to_csv(OUTPUT / "selected.csv", index=False)
    combined = pd.concat([*baseline_frames, predictions], ignore_index=True)
    combined.to_csv(OUTPUT / "development_predictions.csv", index=False)
    pd.concat(all_details, ignore_index=True).to_csv(OUTPUT / "fold_metrics.csv", index=False)
    export_metrics(combined, "development", OUTPUT)
    train_rows = predictions[predictions.split == "train"]
    pd.crosstab(train_rows.cluster, train_rows.label).reindex(columns=CLASSES, fill_value=0).assign(
        mapped_label=pd.Series(bundle["mapping"])
    ).to_csv(OUTPUT / "cluster_train_counts.csv")
    plot_confusions(combined, "cv")
    plot_confusions(combined, "validation")
    print(f"Selected config {row['id']} | promoted: {promote}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        study()
