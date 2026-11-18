"""Compare the existing model with compact descriptors and field-photo training."""

import argparse
from dataclasses import asdict, replace

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from dataset_preparation import CLASSES, RESULTS, SEED
from generalization_data import FIELD, FRESH, OUTPUT
from generalization_features import compact_cache
from feature_extraction import cached_features
from train_models import FILES, MODELS, Experiment, accuracy_and_f1, combine, fit_model, predict

STUDY_MODELS = OUTPUT / "models"
SOURCES = ("kaggle", "field")
PLAIN = ("global_color", "global_shape")
FOREGROUND = (*PLAIN, "foreground_color", "foreground_shape")
JOINT_TRAINING = ("mixed", "balanced", "balanced_classes")


def partition():
    kaggle = pd.read_csv(RESULTS / "manifest.csv")
    kaggle = kaggle[kaggle.split.isin(["train", "validation"])].assign(source="kaggle")
    kaggle["group"] = "k:" + kaggle.group.astype(str)
    original_folds = pd.read_csv(RESULTS / "cv_folds.csv").set_index("path").fold
    kaggle["fold"] = kaggle.path.map(original_folds)

    field = pd.read_csv(FIELD).assign(source="field", split="train")
    field["group"] = "f:" + field.observer
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, held = next(splitter.split(field, field.label, field.group))
    field.loc[held, "split"] = "validation"
    fitting = field[field.split == "train"]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for i, (_, fold) in enumerate(splitter.split(fitting, fitting.label, fitting.group), 1):
        field.loc[fitting.index[fold], "fold"] = i
    dev = pd.concat([kaggle, field], ignore_index=True)
    assert dev.groupby("group").split.nunique().max() == 1
    assert dev.groupby("group").fold.nunique().max() == 1
    assert dev.groupby(["source", "split"]).label.nunique().eq(5).all()
    assert dev[dev.split == "train"].groupby(["source", "fold"]).label.nunique().eq(5).all()
    dev[["path", "label", "source", "group", "split", "fold"]].to_csv(
        OUTPUT / "development_split.csv", index=False
    )
    return dev


def parts_for(rows, experiment):
    mode = experiment.mode
    if mode.startswith("compact-"):
        return compact_cache(rows, mode, OUTPUT / "cache")
    parts = cached_features(rows, mode)
    if "local" in experiment.groups:
        parts.update(cached_features(rows, f"local-{mode}"))
    return parts


def train(dev, mode, experiment, x, dimensions, indices):
    fold = dev.iloc[indices]
    if mode == "kaggle":
        mask = fold.source.eq("kaggle").to_numpy()
        indices, fold = indices[mask], fold[mask]
    elif mode == "field":
        mask = fold.source.eq("field").to_numpy()
        indices, fold = indices[mask], fold[mask]
    weight = None
    if mode in ("balanced", "balanced_classes"):
        strata = fold.source if mode == "balanced" else fold.source + ":" + fold.label
        counts = strata.value_counts()
        weight = strata.map(len(fold) / (len(counts) * counts)).to_numpy()
    config = replace(experiment, components=min(experiment.components, len(fold)))
    return fit_model(config, dimensions, x[indices], fold.label.to_numpy(dtype=str), weight)


def predictions(dev, indices, name, split, bundle, x):
    return dev.iloc[indices][["path", "source", "label"]].assign(
        candidate=name, split=split, predicted=predict(bundle, x[indices])
    )


def score(frame):
    accuracy, f1 = accuracy_and_f1(frame.label, frame.predicted)
    return {"accuracy": accuracy, "macro_f1": f1, "n": len(frame)}


def compare(dev, experiment, training, x, dimensions):
    fit = np.flatnonzero(dev.split == "train")
    validation = np.flatnonzero(dev.split == "validation")
    folds = []
    for i in range(1, 4):
        a = fit[dev.fold.iloc[fit] != i]
        b = fit[dev.fold.iloc[fit] == i]
        folds.append(predictions(dev, b, experiment.name, f"cv_{i}",
                                 train(dev, training, experiment, x, dimensions, a), x))
    bundle = train(dev, training, experiment, x, dimensions, fit)
    training_rows = fit if training in JOINT_TRAINING else fit[
        dev.source.iloc[fit] == training
    ]
    outputs = pd.concat([*folds, predictions(dev, training_rows, experiment.name, "train", bundle, x),
                         predictions(dev, validation, experiment.name, "validation", bundle, x)])
    row = {**asdict(experiment), "training": training,
           "estimator_dimensions": bundle["pipeline"].named_steps["estimator"].n_features_in_}
    for source in SOURCES:
        subset = outputs[outputs.source == source]
        for split in ("train", "cv", "validation"):
            part = subset[subset.split.str.startswith(split)]
            if len(part):
                for metric in ("accuracy", "macro_f1"):
                    values = [score(group)[metric] for _, group in part.groupby("split")]
                    row[f"{source}_{split}_{metric}"] = np.mean(values)
    row["selection_f1"] = np.mean([row[f"{source}_cv_macro_f1"] for source in SOURCES])
    print(f"{experiment.name}: Kaggle CV {row['kaggle_cv_accuracy']:.2%}, "
          f"field CV {row['field_cv_accuracy']:.2%}, selection F1 {row['selection_f1']:.2%}",
          flush=True)
    return row, bundle, outputs


def candidates(baseline):
    plain = Experiment("Compact crop C=1", mode="compact-crop", groups=PLAIN,
                       weights=(1.5, 1.), scale="standard", c=1.)
    full = replace(plain, name="Compact full frame C=1", mode="compact-pad")
    fg = replace(full, name="Foreground full frame C=1", groups=FOREGROUND,
                 weights=(1., .6, 1.6, 1.))
    return [
        (baseline, "kaggle"),
        (replace(baseline, name="Nystrom full frame", mode="pad"), "kaggle"),
        (replace(baseline, name="Nystrom C=1", c=1.), "kaggle"),
        (plain, "kaggle"), (full, "kaggle"),
        (replace(full, name="Compact full frame C=0.1", c=.1), "kaggle"),
        (fg, "kaggle"),
        (replace(fg, name="Foreground crop C=1", mode="compact-crop"), "kaggle"),
        (replace(full, name="Compact mixed C=1"), "mixed"),
        (replace(fg, name="Foreground mixed C=1"), "mixed"),
        (replace(fg, name="Foreground mixed balanced C=0.1", c=.1), "balanced"),
        (replace(fg, name="Foreground mixed balanced C=1"), "balanced"),
        (replace(fg, name="Foreground crop balanced C=1", mode="compact-crop"), "balanced"),
        (replace(fg, name="Foreground mixed balanced C=5", c=5.), "balanced"),
        (replace(fg, name="Foreground source and class balanced"), "balanced_classes"),
        (replace(fg, name="Foreground field only C=1"), "field"),
        (replace(baseline, name="Nystrom mixed"), "mixed"),
        (replace(baseline, name="Nystrom mixed balanced"), "balanced"),
        (replace(fg, name="KNN compact mixed", model="KNN Classifier", pca=64,
                 neighbors=15, vote="uniform"), "mixed"),
        (replace(fg, name="K-Means compact mixed", model="K-Means", pca=32,
                 one_to_one=True), "mixed"),
    ]


def export_metrics(predicted, prefix, directory=OUTPUT):
    rows, per_class, matrices = [], [], []
    keys = ["candidate", "source", "split"]
    predicted = predicted.assign(split=predicted.split.str.replace(r"^cv_\d+$", "cv", regex=True))
    for (name, source, split), group in predicted.groupby(keys):
        key = dict(candidate=name, source=source, split=split)
        rows.append({**key, **score(group)})
        report = classification_report(group.label, group.predicted, labels=CLASSES,
                                       zero_division=0, output_dict=True)
        per_class.extend({**key, "label": label, **report[label]} for label in (*CLASSES, "macro avg"))
        matrix = confusion_matrix(group.label, group.predicted, labels=CLASSES)
        for i, label in enumerate(CLASSES):
            matrices.extend({**key, "actual": label, "predicted": outcome, "count": matrix[i, j],
                             "row_percent": 100 * matrix[i, j] / max(matrix[i].sum(), 1)}
                            for j, outcome in enumerate(CLASSES))
    for suffix, data in (("metrics", rows), ("per_class", per_class), ("confusions", matrices)):
        pd.DataFrame(data).to_csv(directory / f"{prefix}_{suffix}.csv", index=False)


def curves(dev, inputs):
    fit = dev[dev.split == "train"]
    splits = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    strata = fit.source + ":" + fit.label
    groups = [fit.index[held].to_numpy() for _, held in splits.split(fit, strata, fit.group)]
    validation = np.flatnonzero(dev.split == "validation")
    rows = []
    for experiment, training, x, dimensions in inputs:
        for count in range(1, 5):
            indices = np.sort(np.concatenate(groups[:count]))
            model = train(dev, training, experiment, x, dimensions, indices)
            train_rows = indices if training in JOINT_TRAINING else indices[
                dev.source.iloc[indices] == training
            ]
            for split, row_ids in (("train", train_rows), ("validation", validation)):
                frames = predictions(dev, row_ids, experiment.name, split, model, x)
                for source, part in frames.groupby("source"):
                    rows.append({"candidate": experiment.name, "source": source, "split": split,
                                 "training_images": len(train_rows), **score(part)})
    data = pd.DataFrame(rows)
    data.to_csv(OUTPUT / "learning_curves.csv", index=False)
    plot_curves(data)


def plot_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = dict(zip(data.candidate.unique(), ("#386e98", "#a55030", "#378153")))
    line_styles = {"train": "--", "validation": "-"}
    for source, ax in zip(SOURCES, axes, strict=True):
        for (candidate, split), group in data[data.source == source].groupby(["candidate", "split"]):
            ax.plot(group.training_images, 100 * group.accuracy, marker="o",
                    color=colors[candidate], linestyle=line_styles[split])
        ax.set(title=source.title(), xlabel="Training images", ylim=(10, 105))
    axes[0].set_ylabel("Accuracy (%)")
    legend = [Line2D([0], [0], color=color, label=name) for name, color in colors.items()]
    legend.extend(Line2D([0], [0], color="#444444", linestyle=style, label=split.title())
                  for split, style in line_styles.items())
    fig.legend(handles=legend, loc="lower center", ncols=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(OUTPUT / "learning_curves.png", dpi=140)
    plt.close(fig)


def development():
    dev = partition()
    old = joblib.load(MODELS / FILES["Logistic Regression"])
    baseline = Experiment(**{**old["experiment"], "name": "Existing Nystrom"})
    experiments, frames, best, curve_inputs, cache = [], [], {}, {}, {}
    compact = None
    for experiment, training in candidates(baseline):
        key = (experiment.mode, "local" in experiment.groups)
        if key not in cache:
            parts = [parts_for(dev[dev.source == source], experiment) for source in SOURCES]
            cache[key] = {group: np.concatenate([p[group] for p in parts]) for group in parts[0]}
        groups = cache[key]
        x = combine(groups, experiment.groups)
        dimensions = [groups[group].shape[1] for group in experiment.groups]
        row, bundle, frame = compare(dev, experiment, training, x, dimensions)
        experiments.append(row)
        frames.append(frame)
        family = experiment.model
        if family not in best or row["selection_f1"] > best[family][0]["selection_f1"]:
            best[family] = (row, bundle)
            curve_inputs[family] = (experiment, training, x, dimensions)
        if family == "Logistic Regression" and experiment.mode.startswith("compact-") and (
            compact is None or row["selection_f1"] > compact[0]["selection_f1"]
        ):
            compact = (row, bundle)
            curve_inputs["compact"] = (experiment, training, x, dimensions)
        if experiment == baseline:
            curve_inputs["baseline"] = (experiment, training, x, dimensions)
        pd.DataFrame(experiments).to_csv(OUTPUT / "experiments.csv", index=False)
    STUDY_MODELS.mkdir(exist_ok=True)
    for family, (_, bundle) in best.items():
        joblib.dump(bundle, STUDY_MODELS / FILES[family], compress=3)
    joblib.dump(compact[1], STUDY_MODELS / "compact_logistic.joblib", compress=3)
    pd.DataFrame([compact[0]]).to_csv(OUTPUT / "selected_compact.csv", index=False)
    pd.DataFrame([row for row, _ in best.values()]).to_csv(OUTPUT / "selected.csv", index=False)
    predicted = pd.concat(frames, ignore_index=True)
    predicted.to_csv(OUTPUT / "development_predictions.csv", index=False)
    export_metrics(predicted, "development")
    curve_names = ("baseline", "Logistic Regression", "compact")
    inputs = {curve_inputs[name][0].name: curve_inputs[name] for name in curve_names}
    curves(dev, list(inputs.values()))


def final_test():
    rows = pd.read_csv(FRESH).assign(source="fresh_iNat_2026")
    selected = pd.read_csv(OUTPUT / "selected.csv").set_index("model")
    models = [(MODELS / FILES["Logistic Regression"], "Existing Nystrom")]
    models.extend((STUDY_MODELS / file, selected.at[family, "name"]) for family, file in FILES.items())
    compact = pd.read_csv(OUTPUT / "selected_compact.csv").iloc[0]["name"]
    if compact != selected.at["Logistic Regression", "name"]:
        models.append((STUDY_MODELS / "compact_logistic.joblib", compact))
    out = []
    for path, name in models:
        bundle = joblib.load(path)
        config = Experiment(**bundle["experiment"])
        x = combine(parts_for(rows, config), config.groups)
        frame = predictions(rows, np.arange(len(rows)), name, "fresh_test", bundle, x)
        print(name, score(frame), flush=True)
        out.append(frame)
    output = pd.concat(out, ignore_index=True)
    output.to_csv(OUTPUT / "fresh_test_predictions.csv", index=False)
    export_metrics(output, "fresh_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-test", action="store_true", help="Score the locked finalists.")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        (final_test if args.fresh_test else development)()
