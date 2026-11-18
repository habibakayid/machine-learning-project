"""Prepare a new field sample with no shared observers or detected image copies."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from dataset_preparation import CLASSES, EXTENSIONS, RAW, RESULTS, ROOT, create_groups, related_pairs, scan_images
from external_dataset import TAXA, download, sample_observations
from generalization_data import FIELD, FRESH
from knn_holdout import HOLDOUT as KNN_HOLDOUT

OUTPUT = RESULTS.parent / "logistic_study"
HOLDOUT = OUTPUT / "fresh_manifest.csv"


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    if HOLDOUT.exists():
        rows = pd.read_csv(HOLDOUT)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for digest, actual in zip(rows.sha256, pool.map(download, rows.to_dict("records")), strict=True):
                if digest != actual["sha256"]:
                    raise ValueError(f"Photo hash changed: {actual['path']}")
        print(f"Verified the {len(rows)} fixed photos.", flush=True)
        return

    previous = pd.concat([pd.read_csv(p) for p in (
        FIELD, FRESH, RESULTS / "external_manifest.csv", KNN_HOLDOUT,
    )])
    blocked = set(previous.observer)

    def sample(item):
        label, taxon = item
        selected = sample_observations(label, taxon, "2016-01-01", "2019-12-31",
                                       OUTPUT / "photos", blocked, goal=160)[:36]
        print(f"{label}: {len(selected)} eligible photos", flush=True)
        return selected

    with ThreadPoolExecutor(max_workers=2) as pool:
        candidates = [row for batch in pool.map(sample, TAXA.items()) for row in batch]
    if {row["label"] for row in candidates} != set(CLASSES):
        raise ValueError("The new sample must cover all five classes.")
    with ThreadPoolExecutor(max_workers=12) as pool:
        fresh = pd.DataFrame(pool.map(download, candidates))

    known = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in EXTENSIONS)
    known.extend(ROOT / path for path in previous.path)
    inventory, images, unreadable = scan_images(
        known + [ROOT / path for path in fresh.path], lambda p: p.parent.name,
        lambda p: p.relative_to(ROOT).as_posix(),
    )
    if unreadable:
        raise ValueError(f"Overlap audit could not read {len(unreadable)} files.")
    groups = create_groups(len(inventory), related_pairs(inventory, images))
    used = set(groups[:len(known)])
    keep = []
    for group in groups[len(known):]:
        keep.append(group not in used)
        used.add(group)
    if set(fresh[keep].label) != set(CLASSES):
        raise ValueError("A class was lost during the overlap audit.")
    fresh[~pd.Series(keep, index=fresh.index)].to_csv(OUTPUT / "overlap_exclusions.csv", index=False)
    fresh[keep].to_csv(HOLDOUT, index=False)
    print(f"Fixed sample: {fresh[keep].label.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    prepare()
