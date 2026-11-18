"""Check exported counts, class metrics, model hashes and recorded GUI coverage."""

import hashlib
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT.parent))
from dataset_preparation import CLASSES


def main():
    required = (
        "FINAL_REPORT.md", "BASELINE_COMPARISON.csv", "FINAL_MODEL_METRICS.csv",
        "FINAL_PER_CLASS_METRICS.csv", "ALL_EXPERIMENTS.csv",
        "FINAL_CONFUSION_MATRICES.csv", "FINAL_DATA_SPLITS.csv",
        "FINAL_MODEL_SELECTION.md", "OVERFITTING_ANALYSIS.md",
        "DATA_INTEGRITY_AUDIT.md", "MODEL_MANIFEST.json",
    )
    assert all((OUT / name).is_file() for name in required)
    metrics = pd.read_csv(OUT / "FINAL_MODEL_METRICS.csv")
    per_class = pd.read_csv(OUT / "FINAL_PER_CLASS_METRICS.csv")
    counts = pd.read_csv(OUT / "FINAL_CONFUSION_MATRICES.csv")
    keys = ["scope", "family", "candidate", "source", "evaluation"]
    checked = 0
    for key, rows in counts.groupby(keys):
        matrix = rows.pivot(index="actual", columns="predicted", values="count").loc[CLASSES, CLASSES].to_numpy()
        percentages = rows.pivot(index="actual", columns="predicted", values="row_percent").loc[CLASSES, CLASSES].to_numpy()
        assert matrix.shape == (5, 5) and (matrix >= 0).all()
        assert np.equal(matrix, matrix.astype(int)).all()
        assert np.allclose(percentages, 100 * matrix / matrix.sum(axis=1, keepdims=True))
        match = metrics.set_index(keys).loc[key]
        assert matrix.sum() == match.n and matrix.trace() == match.correct
        assert np.isclose(matrix.trace() / matrix.sum(), match.pooled_accuracy)
        class_rows = per_class.set_index(keys).loc[key].set_index("label").loc[list(CLASSES)]
        precision = np.divide(np.diag(matrix), matrix.sum(axis=0),
                              out=np.zeros(5), where=matrix.sum(axis=0) > 0)
        recall = np.diag(matrix) / matrix.sum(axis=1)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros(5), where=precision + recall > 0)
        assert np.allclose(class_rows.precision, precision)
        assert np.allclose(class_rows.recall, recall)
        assert np.allclose(class_rows["f1-score"], f1)
        if key[-1] not in ("cv", "group_excluded_refitted_3fold"):
            assert np.isclose(match.macro_f1, f1.mean())
        checked += 1

    manifest = json.loads((OUT / "MODEL_MANIFEST.json").read_text())
    originals = json.loads((OUT / "baseline_snapshot.json").read_text())
    for item in [*manifest, *originals]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    for item in manifest:
        bundle = joblib.load(ROOT / item["path"])
        estimator = bundle["pipeline"].steps[-1][1]
        assert item["class_order"] == list(CLASSES)
        if item["scope"] == "primary":
            assert item["fit_counts"] == {"kaggle": 3017}
        else:
            assert item["fit_counts"] == {"field": 450, "kaggle": 3017}
        if bundle["mapping"] is not None:
            assert set(bundle["mapping"]) == set(CLASSES)
            assert len(bundle["mapping"]) == estimator.n_clusters
        else:
            assert list(estimator.classes_) == list(CLASSES)

    gui = json.loads((OUT / "GUI_TEST_RESULTS.json").read_text())
    cases = pd.DataFrame(gui["cases"])
    assert len(cases) == 30 and cases.result.eq("passed").all()
    for _, group in cases.groupby(["scope", "family"]):
        assert set(group.actual) == set(CLASSES)
        assert set(group.predicted) == set(CLASSES)
    assert not gui["console_errors"]
    for name in ("corrupt_upload", "unsupported_format", "recovery", "mobile_layout"):
        assert gui[name] == "passed"

    for document in [ROOT / "README.md", *OUT.glob("*.md")]:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" not in target and not target.startswith("#"):
                assert (document.parent / target.split("#")[0]).exists(), (document, target)
    result = {
        "required_deliverables": len(required), "metric_partitions_recomputed": checked,
        "selected_model_hashes_verified": len(manifest), "original_model_hashes_unchanged": len(originals),
        "browser_cases_verified": len(cases), "five_label_output_coverage": "all six suites/models",
        "local_document_links": "passed", "result": "passed",
    }
    (OUT / "VERIFICATION_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
