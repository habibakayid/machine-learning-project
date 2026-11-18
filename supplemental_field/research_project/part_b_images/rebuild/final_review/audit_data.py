"""Recheck the fixed partitions and cross-cohort image and observer overlap."""

import json
from itertools import combinations
from pathlib import Path
import sys

import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from dataset_preparation import (CLASSES, EXTENSIONS, RAW, RESULTS, ROOT,
                                 create_groups, related_pairs, scan_images)

GENERAL = OUT.parent / "generalization"
EXTERNAL = {
    "field_2025_234": RESULTS / "external_manifest.csv",
    "field_2026_150": GENERAL / "fresh_manifest.csv",
    "field_2020_2021_191": OUT.parent / "knn_study" / "fresh_manifest.csv",
    "field_2016_2019_156": OUT.parent / "logistic_study" / "fresh_manifest.csv",
}


def audit_data():
    kaggle = pd.read_csv(RESULTS / "manifest.csv")
    raw = pd.read_csv(RESULTS / "raw_inventory.csv")
    folds = pd.read_csv(RESULTS / "cv_folds.csv")
    assert set(folds.path) == set(kaggle.loc[kaggle.split == "train", "path"])
    assert folds.path.is_unique and folds.groupby("group").fold.nunique().max() == 1
    dev = pd.read_csv(GENERAL / "development_split.csv")
    field_rows = dev[dev.source.eq("field")][["path", "split", "group", "fold"]]
    field = pd.read_csv(GENERAL / "field_manifest.csv").merge(field_rows, on="path",
                                                             validate="one_to_one")
    assert set(field.path) == set(field_rows.path)
    assert field.groupby("observer").split.nunique().max() == 1
    assert field.groupby("observer").fold.nunique().max() == 1
    assert kaggle.groupby("group").split.nunique().max() == 1
    assert dev.groupby("group").split.nunique().max() == 1
    assert dev.groupby("group").fold.nunique().max() == 1
    original = folds.merge(dev[dev.source.eq("kaggle")], on="path", suffixes=("_old", "_dev"))
    assert len(original) == len(folds) and original.fold_old.eq(original.fold_dev).all()
    assert set(dev[dev.source.eq("kaggle") & dev.split.eq("validation")].path) == set(
        kaggle[kaggle.split.eq("validation")].path)
    assert dev[dev.split.eq("train")].groupby(["source", "fold"]).label.nunique().eq(5).all()

    cohorts = {f"kaggle_{split}": rows for split, rows in kaggle.groupby("split")}
    cohorts.update({f"field_{split}": rows for split, rows in field.groupby("split")})
    cohorts.update({name: pd.read_csv(path) for name, path in EXTERNAL.items()})
    legacy_dir = ROOT / "part_b_images" / "data" / "processed" / "splits"
    legacy = {f"legacy_kaggle_{split}": pd.read_csv(legacy_dir / f"{split}.csv")
              for split in ("train", "validation", "test")}
    partitions = {"kaggle_raw": raw, **cohorts, **legacy}
    for (source, fold), rows in dev[dev.split.eq("train")].groupby(["source", "fold"]):
        partitions[f"{source}_cv_fold_{int(fold)}"] = rows
    counts = []
    for name, rows in partitions.items():
        values = rows.label.value_counts()
        assert set(values.index) == set(CLASSES)
        counts.append({"source": "Kaggle" if "kaggle" in name else "iNaturalist", "partition": name,
                       **{label: values[label] for label in CLASSES}, "total": len(rows)})
    pd.DataFrame(counts).to_csv(OUT / "DATA_SPLITS.csv", index=False)

    paths = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in EXTENSIONS)
    paths.extend(ROOT / p for name in EXTERNAL for p in cohorts[name].path)
    paths.extend(ROOT / p for p in field.path)
    assert len(paths) == len(set(paths))
    inventory, images, unreadable = scan_images(
        paths, lambda p: p.parent.name, lambda p: p.relative_to(ROOT).as_posix())
    pairs = list(set(related_pairs(inventory, images)))
    inventory["related_group"] = create_groups(len(inventory), pairs)
    inventory = inventory.set_index("path")
    for rows in [raw, kaggle, field, *(cohorts[k] for k in EXTERNAL)]:
        disk = inventory.reindex(rows.path)
        assert disk.sha256.tolist() == rows.sha256.tolist()
        assert disk.label.tolist() == rows.label.tolist()

    overlaps = []
    active = {**cohorts, **legacy}
    for first, second in combinations(active, 2):
        a, b = active[first], active[second]
        ai, bi = inventory.loc[a.path], inventory.loc[b.path]
        observers = len(set(a.observer) & set(b.observer)) if "observer" in a and "observer" in b else None
        overlaps.append({
            "first": first, "second": second, "same_paths": len(set(a.path) & set(b.path)),
            "shared_byte_hashes": len(set(ai.sha256) & set(bi.sha256)),
            "shared_pixel_hashes": len(set(ai.pixel_sha256) & set(bi.pixel_sha256)),
            "shared_related_groups": len(set(ai.related_group) & set(bi.related_group)),
            "shared_observers": observers,
        })
    output = pd.DataFrame(overlaps)
    output.to_csv(OUT / "DATA_OVERLAP.csv", index=False)
    old = pd.concat([rows.assign(split=name) for name, rows in legacy.items()])

    def crossing(rows):
        groups = inventory.loc[rows.path].related_group.to_numpy()
        return int((rows.assign(related=groups).groupby("related").split.nunique() > 1).sum())

    summary = {
        "scanned": len(inventory), "unreadable": unreadable, "raw_count": len(raw),
        "retained_kaggle": len(kaggle), "kaggle_groups": kaggle.group.nunique(),
        "legacy_related_groups_crossing_splits": crossing(old),
        "rebuilt_related_groups_crossing_splits": crossing(kaggle),
        "related_groups_crossing_cv_folds": crossing(
            dev[dev.split.eq("train")].assign(split=dev.fold)),
        "active_cross_partition_matches": output[
            ~output["first"].str.startswith("legacy") & ~output["second"].str.startswith("legacy")
            & output.shared_related_groups.gt(0)
        ].to_dict("records"),
    }
    (OUT / "DATA_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    audit_data()
