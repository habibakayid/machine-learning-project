"""Controlled primary Kaggle and supplemental mixed-source development experiments."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from itertools import product
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from dataset_preparation import CLASSES, ROOT, SEED
from generalization_study import parts_for
from knn_study import (Geometry, development_rows, fit_space, knn, vote_labels)
from kmeans_study import cluster_labels, metrics as cluster_metrics
from logistic_study import fit_weights
from train_models import FILES, MODELS, Experiment, combine, make_pipeline

CS = (.1, .3, .5, 1., 2., 3., 5., 10.)
KS = (3, 5, 7, 9, 11, 15, 21, 31)
BASELINES = {
    "primary": {name: MODELS / file for name, file in FILES.items()},
    "supplemental": {
        "Logistic Regression": OUT.parent / "generalization/models/logistic_regression.joblib",
        "KNN Classifier": OUT.parent / "knn_study/knn_classifier.joblib",
        "K-Means": OUT.parent / "kmeans_study/promoted_kmeans.joblib",
    },
}
POLICY = """# Selection policy and documented amendment

Primary fits Kaggle training rows only, and uses Kaggle CV macro F1.
Supplemental fits the existing Kaggle and field training rows and averages
their CV macro F1 with equal weight. All three fold assignments are retained.
Candidates must emit all five labels in the final mapping.

Replacement requires at least 0.002 improvement in the CV selection score,
no more than 0.005 loss in validation macro F1, and no more than 0.01 loss
in validation accuracy in any source. For each class in each source, pooled
CV and validation recall and F1 may fall by at most 0.10.
The weakest fold's macro F1 in each source may fall by at most 0.01.
Fold standard deviations are reported alongside that absolute performance check.
Logistic Regression must also reduce both sources' train-validation accuracy
gaps (Kaggle only for the primary experiment), preserving the earlier gap rule.
Among eligible candidates choose the highest CV score, then the smaller fold
F1 spread. Every failed guardrail is saved. The baseline remains eligible.

Validation is an already-accessed development confirmation set. All historical
tests remain historical, and are excluded from this experiment runner.
Numerical target ranges do not enter model selection.
"""


def snapshot():
    OUT.mkdir(exist_ok=True)
    target = OUT / "baseline_snapshot.json"
    if target.exists():
        return
    files = sorted(OUT.parent.rglob("*.joblib"))
    records = [{"path": p.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    target.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    prior = OUT.parent / "final_review/FINAL_METRICS.csv"
    pd.read_csv(prior).to_csv(OUT / "BASELINE_COMPARISON.csv", index=False)
    (OUT / "SELECTION_POLICY.md").write_text(POLICY, encoding="utf-8")


def scores(actual, predicted):
    report = classification_report(actual, predicted, labels=CLASSES, zero_division=0,
                                   output_dict=True)
    return {"accuracy": np.mean(actual == predicted),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"]}, report


class Study:
    def __init__(self, scope, family):
        self.scope = scope
        self.family = family
        self.dev = development_rows()
        if scope == "primary":
            self.dev = self.dev[self.dev.source.eq("kaggle")].reset_index(drop=True)
        self.sources = self.dev.source.unique().tolist()
        self.y = self.dev.label.to_numpy(dtype=str)
        self.fit = np.flatnonzero(self.dev.split.eq("train"))
        self.val = np.flatnonzero(self.dev.split.eq("validation"))
        self.folds = [(self.fit[self.dev.fold.iloc[self.fit] != i],
                       self.fit[self.dev.fold.iloc[self.fit] == i]) for i in (1, 2, 3)]
        for a, b in self.folds:
            assert not set(self.dev.group.iloc[a]) & set(self.dev.group.iloc[b])
        assert not set(self.dev.group.iloc[self.fit]) & set(self.dev.group.iloc[self.val])
        self.folder = OUT / scope / family.replace(" ", "_").lower()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.rows, self.predictions, self.class_rows = [], [], []
        self.baseline = None
        self.selected = None
        self.best_cv = None
        self.final_bundle = None
        self.base_bundle = joblib.load(BASELINES[scope][family])
        self.exp = Experiment(**self.base_bundle["experiment"])

    def data(self, exp):
        key = exp.mode, "local" in exp.groups
        if key not in self.cache:
            chunks = [parts_for(self.dev[self.dev.source.eq(s)], exp) for s in self.sources]
            self.cache[key] = {g: np.concatenate([chunk[g] for chunk in chunks])
                               for g in chunks[0]}
        parts = self.cache[key]
        return combine(parts, exp.groups), [parts[g].shape[1] for g in exp.groups]

    def frame(self, ids, prediction, split, **extra):
        return self.dev.iloc[ids][["path", "source", "label", "group", "fold"]].assign(
            predicted=prediction, evaluation=split, **extra)

    def guardrails(self, metrics, per_class):
        if self.baseline is None:
            return []
        reference, ref_classes = self.baseline
        failures = []
        if metrics["selection_f1"] < reference["selection_f1"] + .002:
            failures.append("CV gain below 0.002")
        for source in self.sources:
            if metrics[f"{source}_cv_min_f1"] < reference[f"{source}_cv_min_f1"] - .01:
                failures.append(f"{source} weakest CV fold")
            for metric, limit in (("macro_f1", .005), ("accuracy", .01)):
                key = f"{source}_validation_{metric}"
                if metrics[key] < reference[key] - limit:
                    failures.append(f"{source} validation {metric}")
            if self.family == "Logistic Regression" and (
                metrics[f"{source}_gap"] >= reference[f"{source}_gap"]):
                failures.append(f"{source} gap did not shrink")
        check = per_class[per_class.evaluation.isin(["cv", "validation"])]
        joined = check.merge(ref_classes, on=["source", "evaluation", "label"],
                             suffixes=("", "_base"))
        for metric in ("recall", "f1-score"):
            for row in joined[joined[metric] < joined[metric + "_base"] - .10].itertuples():
                failures.append(f"{row.source} {row.evaluation} {row.label} {metric}")
        return failures

    def record(self, name, stage, exp, frames, bundle, **settings):
        frame = pd.concat(frames, ignore_index=True).assign(
            candidate=name, scope=self.scope, family=self.family)
        values = []
        for (source, split), group in frame.groupby(["source", "evaluation"]):
            value, report = scores(group.label.to_numpy(), group.predicted.to_numpy())
            values.append({"source": source, "evaluation": split, **value})
        metric = {"scope": self.scope, "family": self.family, "candidate": name, "stage": stage,
                  "c": exp.c if self.family == "Logistic Regression" else None,
                  "pca": exp.pca, "components": exp.components if exp.kernel else 0,
                  "gamma": exp.gamma if exp.kernel else None,
                  "groups": exp.groups, "weights": exp.weights, "scale": exp.scale,
                  **settings}
        score_table = pd.DataFrame(values)
        for source in self.sources:
            for split in ("train", "validation"):
                row = score_table[score_table.source.eq(source) & score_table.evaluation.eq(split)].iloc[0]
                for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                    metric[f"{source}_{split}_{key}"] = row[key]
            fold_values = score_table[score_table.source.eq(source) &
                                      score_table.evaluation.str.startswith("cv_")]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                metric[f"{source}_cv_{key}"] = fold_values[key].mean()
            metric[f"{source}_cv_f1_std"] = fold_values.macro_f1.std(ddof=0)
            metric[f"{source}_cv_min_f1"] = fold_values.macro_f1.min()
            metric[f"{source}_gap"] = metric[f"{source}_train_accuracy"] - metric[f"{source}_validation_accuracy"]
        metric["selection_f1"] = np.mean([metric[f"{s}_cv_macro_f1"] for s in self.sources])
        metric["cv_f1_std"] = np.mean([metric[f"{s}_cv_f1_std"] for s in self.sources])
        frame["evaluation"] = frame.evaluation.str.replace(r"^cv_\d+$", "cv", regex=True)
        classes = []
        for (source, split), rows in frame.groupby(["source", "evaluation"]):
            _, report = scores(rows.label.to_numpy(), rows.predicted.to_numpy())
            classes.extend({"source": source, "evaluation": split, "label": label, **report[label]}
                           for label in CLASSES)
        per_class = pd.DataFrame(classes)
        failures = self.guardrails(metric, per_class)
        if bundle["mapping"] is not None and set(bundle["mapping"]) != set(CLASSES):
            failures.append("mapping misses a class")
        metric["eligible"] = not failures
        metric["guardrail_failures"] = " | ".join(failures)
        if name == "baseline":
            self.baseline = metric, per_class
            self.selected = metric
            self.final_bundle = bundle
        elif metric["eligible"] and metric["selection_f1"] > self.selected["selection_f1"]:
            self.selected = metric
            self.final_bundle = bundle
        if self.best_cv is None or metric["selection_f1"] > self.best_cv["selection_f1"]:
            self.best_cv = metric
            joblib.dump(bundle, self.folder / "best_cv_candidate.joblib", compress=3)
        self.rows.append(metric)
        self.predictions.append(frame)
        self.class_rows.append(per_class.assign(candidate=name, scope=self.scope, family=self.family))
        pd.DataFrame(self.rows).to_csv(self.folder / "experiments.csv", index=False)
        print(f"{self.scope} / {self.family} / {name} | CV F1 {metric['selection_f1']:.2%}"
              f" | eligible={metric['eligible']}", flush=True)

    def finish(self):
        joblib.dump(self.final_bundle, self.folder / "selected.joblib", compress=3)
        pd.DataFrame([self.selected]).to_csv(self.folder / "selected.csv", index=False)
        pd.DataFrame([self.best_cv]).to_csv(self.folder / "best_cv.csv", index=False)
        predictions = pd.concat(self.predictions, ignore_index=True)
        predictions.to_csv(self.folder / "predictions.csv.gz", index=False, compression="gzip")
        pd.concat(self.class_rows).to_csv(self.folder / "per_class.csv", index=False)
        print(f"FINAL {self.scope} / {self.family}: {self.selected['candidate']}", flush=True)


def lr_batch(study, exp, variants, stage):
    x, dims = study.data(exp)
    outputs = {name: [] for name, _, _ in variants}
    bundles = {}
    for fold, (fit, held) in enumerate([*study.folds, (study.fit, study.val)], 1):
        transform = make_pipeline(exp, dims)[:-1]
        fit_x = transform.fit_transform(x[fit], study.y[fit])
        held_x = transform.transform(x[held])
        for name, c, weights in variants:
            estimator = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
            estimator.fit(fit_x, study.y[fit], sample_weight=fit_weights(study.dev.iloc[fit], weights))
            outputs[name].append(study.frame(held, estimator.predict(held_x),
                                             "validation" if fold == 4 else f"cv_{fold}"))
            if fold == 4:
                outputs[name].append(study.frame(fit, estimator.predict(fit_x), "train"))
                bundles[name] = {"pipeline": Pipeline([*transform.steps, ("estimator", estimator)]),
                                  "mapping": None, "dimensions": dims,
                                  "experiment": asdict(replace(exp, c=c, name=name))}
    for name, c, weights in variants:
        study.record(name, stage, replace(exp, c=c), outputs[name], bundles[name],
                     sample_weights=weights)


def logistic(study):
    base = study.exp
    lr_batch(study, base, [("baseline", base.c, "mixed"), *[(f"C={c:g}", c, "mixed") for c in CS]],
             "regularization")
    best = study.best_cv
    c = best["c"]
    for components in (600, 900):
        lr_batch(study, replace(base, components=components),
                 [(f"Landmarks={components},C={value:g}", value, "mixed")
                  for value in sorted(set([c, 1.]))], "landmarks")
    best = study.best_cv
    anchor = replace(base, components=int(best["components"]))
    c = best["c"]
    for label, omit in (("No HOG", {"hog"}), ("No LBP", {"lbp"}),
                         ("No edges", {"edges"}), ("No local", {"local"})):
        kept = [(g,w) for g,w in zip(anchor.groups, anchor.weights) if g not in omit]
        groups, weights = zip(*kept)
        lr_batch(study, replace(anchor, groups=groups, weights=weights),
                 [(label, c, "mixed")], "ablation")
    compact = Experiment(**joblib.load(OUT.parent/"generalization/models/compact_logistic.joblib")["experiment"])
    lr_batch(study, compact, [("Compact foreground", 1., "mixed")], "foreground benchmark")
    no_foreground = replace(compact, groups=("global_color", "global_shape"), weights=(1., .6))
    lr_batch(study, no_foreground, [("Compact global only", 1., "mixed")], "foreground ablation")
    if study.scope == "supplemental":
        # The geometry is kept fixed across the weighting comparison.
        for_c = sorted(set([1., c]))
        lr_batch(study, anchor, [(f"{mode},C={value:g}", value, mode) for value in for_c
                                for mode in ("balanced", "balanced_classes")], "source weights")
    study.finish()


def knn_batch(study, geometry, name, variants):
    exp = geometry.features
    x, dims = study.data(exp)
    output = {key: [] for key in variants}
    bundles = {}
    source = study.dev.source.to_numpy(dtype=str)
    for fold, (fit, held) in enumerate([*study.folds, (study.fit, study.val)], 1):
        transform, pool, fitted = fit_space(geometry, dims, x, study.y, source, fit)
        assert np.array_equal(pool, fit)
        held_x = transform.transform(x[held])
        encoded = np.searchsorted(CLASSES, study.y[pool])
        tables = {}
        for metric in {key[0] for key in variants}:
            estimator = knn(metric, max(k for m,k,v in variants if m == metric), "uniform")
            estimator.fit(fitted, study.y[pool])
            tables[metric] = estimator.kneighbors(held_x)
        for metric, k, vote in variants:
            distances, neighbors = tables[metric]
            pred = vote_labels(distances[:, :k], encoded[neighbors[:, :k]], vote)
            key = metric, k, vote
            output[key].append(study.frame(held, pred, "validation" if fold == 4 else f"cv_{fold}"))
            if fold == 4:
                estimator = knn(metric,k,vote).fit(fitted, study.y[pool])
                output[key].append(study.frame(fit, estimator.predict(fitted), "train"))
                config = replace(exp, metric=metric, neighbors=k, vote=vote)
                bundles[key] = {"pipeline": Pipeline([*transform.steps, ("estimator", estimator)]),
                                "mapping": None, "dimensions": dims, "experiment": asdict(config)}
    for key in variants:
        metric,k,vote = key
        config = replace(exp, metric=metric, neighbors=k, vote=vote)
        tag = "baseline" if name == "baseline" else f"{name}/{metric}/k={k}/{vote}"
        study.record(tag, name, config, output[key], bundles[key],
                     projector=geometry.projector, balance_fit=geometry.fit_balance,
                     neighbors=k, metric=metric, vote=vote)


def neighbors(study):
    base = study.exp
    if study.scope == "primary":
        archived = Geometry(base)
        rich = replace(base, scale="standard", kernel="poly", gamma=.18, components=500, pca=0)
        geometry = Geometry(rich, projector="lda")
    else:
        archived = Geometry(base, projector="lda", fit_balance=True)
        geometry = archived
    knn_batch(study, archived, "baseline", [(base.metric,base.neighbors,base.vote)])
    variants = list(product(("manhattan","euclidean"), KS, ("uniform","distance")))
    knn_batch(study, geometry, "neighbor sweep", variants)
    best = study.best_cv
    chosen = [(best["metric"],int(best["neighbors"]),best["vote"])]
    exp = geometry.features
    for label, omit in (("No HOG", {"hog"}), ("No local", {"local"})):
        groups, weights = zip(*[(g,w) for g,w in zip(exp.groups,exp.weights) if g not in omit])
        knn_batch(study, replace(geometry, features=replace(exp, groups=groups, weights=weights)),
                  label, chosen)
    # PCA versus LDA at one fixed neighborhood configuration.
    pca_exp = replace(exp, pca=128, kernel="", scale="standard")
    knn_batch(study, Geometry(pca_exp), "PCA128", chosen)
    compact = Experiment(**joblib.load(OUT.parent/"generalization/models/compact_logistic.joblib")["experiment"])
    knn_batch(study, Geometry(replace(compact, pca=128)), "Compact color texture PCA", chosen)
    study.finish()
    # These diagnostics freeze the fitted representation and omit query neighbors.
    # Full-transform, group-excluded evaluation is the saved fold-refitted CV.
    bundle = study.final_bundle
    pipe = bundle["pipeline"]
    config = Experiment(**bundle["experiment"])
    x, _ = study.data(config)
    pool = study.fit
    coords = pipe[:-1].transform(x[pool])
    estimator = pipe.named_steps["estimator"]
    groups = study.dev.group.to_numpy(dtype=str)[pool]
    max_group = pd.Series(groups).value_counts().max()
    distance, nearest = estimator.kneighbors(coords, n_neighbors=estimator.n_neighbors + max_group)
    encoded = np.searchsorted(CLASSES, study.y[pool])
    diagnostics = []
    for name, allowed in (
        ("leave_one_out_frozen_representation", nearest != np.arange(len(pool))[:,None]),
        ("group_excluded_frozen_representation", groups[nearest] != groups[:,None]),
    ):
        retained = np.argsort(~allowed, axis=1, kind="stable")[:, :estimator.n_neighbors]
        distances = np.take_along_axis(distance,retained,axis=1)
        neighbors = np.take_along_axis(nearest,retained,axis=1)
        pred = vote_labels(distances,encoded[neighbors],estimator.weights)
        diagnostics.append(study.frame(pool,pred,name))
    cv = pd.concat(study.predictions)
    refitted = cv[cv.candidate.eq(study.selected["candidate"]) & cv.evaluation.eq("cv")].copy()
    refitted["evaluation"] = "group_excluded_refitted_3fold"
    pd.concat([*diagnostics,refitted]).to_csv(study.folder/"training_diagnostics.csv",index=False)


def clustering(study):
    base = study.exp
    # The primary baseline uses its original K=5 global representation.
    # The local representation is the previously selected unsupervised geometry.
    local = Experiment(**joblib.load(OUT.parent/"kmeans_study/promoted_kmeans.joblib")["experiment"])
    configurations = [(base,study.base_bundle["pipeline"].named_steps["estimator"].n_clusters,
                       study.base_bundle["pipeline"].named_steps["estimator"].n_init,"baseline")]
    configurations.extend((local,k,10,f"Local PCA20 clusters={k}") for k in (5,10,15))
    diagnostics=[]
    for exp,k,starts,name in configurations:
        x,dims = study.data(exp)
        frames=[]
        bundle=None
        for i,(fit,held) in enumerate([*study.folds,(study.fit,study.val)],1):
            transform=make_pipeline(exp,dims)[:-1]
            fitted=transform.fit_transform(x[fit])
            evaluated=transform.transform(x[held])
            estimator=KMeans(n_clusters=k,n_init=starts,init="k-means++",random_state=SEED)
            estimator.fit(fitted)
            mapping=cluster_labels(estimator.labels_,study.y[fit],k)
            split="validation" if i==4 else f"cv_{i}"
            clusters=estimator.predict(evaluated)
            held_frame=study.frame(held,mapping[clusters],split,cluster=clusters)
            frames.append(held_frame)
            to_score=[(held_frame,evaluated)]
            if i==4:
                fit_frame=study.frame(fit,mapping[estimator.labels_],"train",cluster=estimator.labels_)
                frames.append(fit_frame)
                to_score.append((fit_frame,fitted))
                bundle={"pipeline":Pipeline([*transform.steps,("estimator",estimator)]),
                        "mapping":mapping,"dimensions":dims,"experiment":asdict(exp)}
            for frame,space in to_score:
                for source in study.sources:
                    mask=frame.source.eq(source).to_numpy()
                    diagnostics.append({"candidate":name,"source":source,"evaluation":frame.evaluation.iloc[0],
                                        "clusters":k,**cluster_metrics(frame[mask],space[mask])})
        study.record(name,"limited clusters",exp,frames,bundle,clusters=k,n_init=starts)
    pd.DataFrame(diagnostics).to_csv(study.folder/"cluster_metrics.csv",index=False)
    study.finish()


def main(scope,family):
    snapshot()
    study=Study(scope,family)
    {"Logistic Regression":logistic,"KNN Classifier":neighbors,"K-Means":clustering}[family](study)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope",choices=BASELINES,required=True)
    parser.add_argument("--family",choices=FILES,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        main(args.scope,args.family)
