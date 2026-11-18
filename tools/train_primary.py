"""Refit only the selected recipes, with fold-local preprocessing and frozen splits.

This packaging entry point is new; original experiment implementations are archived.
It has not been executed during packaging. clone resets every fitted transformation.
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from common import CLASSES, FILES, ROOT, development, features, new_output


def evaluate(y, predicted, split):
    report = classification_report(y, predicted, labels=CLASSES, output_dict=True, zero_division=0)
    return {"split": split, "accuracy": float(np.mean(y == predicted)),
            "macro_f1": report["macro avg"]["f1-score"], "per_class": report,
            "confusion": confusion_matrix(y, predicted, labels=CLASSES).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="A NEW output directory; saved release bundles are never overwritten.")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and clone model recipes without fitting or creating output.")
    args = parser.parse_args()
    rows, fit, held, folds = development()
    # Check required inputs before creating a training output folder.
    for path in rows.path:
        if not (ROOT / "primary" / path).is_file():
            raise FileNotFoundError("Import the Kaggle dataset before training.")
    if args.check_only:
        for filename in FILES.values():
            clone(joblib.load(ROOT / "app/models" / filename)["pipeline"])
        print("PASS: Kaggle-only frozen partitions and cloneable selected recipes; no fitting executed.")
        return
    output = new_output(args.output)
    labels = rows.label.to_numpy(dtype=str)
    summaries = []
    for family, filename in FILES.items():
        template = joblib.load(ROOT / "app/models" / filename)
        x = features(rows, template)
        fold_metrics = []
        predictions = labels.copy()
        for index, (training, validation) in enumerate([*folds, (fit, held)]):
            pipe = clone(template["pipeline"])
            assert all(not hasattr(step, "classes_") for _, step in pipe.steps)
            if family == "K-Means":
                pipe.fit(x[training])
                clusters = pipe.predict(x[training])
                mapping = np.asarray(CLASSES)[np.array([
                    [np.count_nonzero((clusters == k) & (labels[training] == name)) for name in CLASSES]
                    for k in range(pipe.named_steps["estimator"].n_clusters)]).argmax(axis=1)]
                predict = lambda ids: mapping[pipe.predict(x[ids])]
            else:
                pipe.fit(x[training], labels[training])
                mapping = None
                predict = lambda ids: pipe.predict(x[ids])
            if index < 3:
                predictions[validation] = predict(validation)
                fold_metrics.append(evaluate(labels[validation], predictions[validation], f"cv_{index+1}"))
            else:
                reports = [evaluate(labels[fit], predict(fit), "train"),
                           evaluate(labels[held], predict(held), "validation"),
                           evaluate(labels[fit], predictions[fit], "cv_pooled")]
                final = {**template, "pipeline": pipe, "mapping": mapping}
                joblib.dump(final, output / filename, compress=3)
        result = {"model": family, "fold_metrics": fold_metrics, "reports": reports,
                  "cv_accuracy": float(np.mean([r["accuracy"] for r in fold_metrics])),
                  "cv_macro_f1": float(np.mean([r["macro_f1"] for r in fold_metrics]))}
        summaries.append(result)
        pd.DataFrame({"path": rows.path.iloc[fit], "label": labels[fit],
                      "cv_prediction": predictions[fit]}).to_csv(output / (filename + ".cv.csv"), index=False)
    (output / "training_results.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
