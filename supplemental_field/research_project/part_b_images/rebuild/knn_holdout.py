"""Collect an unseen field sample without using predictions or changing training data."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from dataset_preparation import CLASSES, EXTENSIONS, RAW, RESULTS, ROOT, create_groups, related_pairs, scan_images
from external_dataset import TAXA, download, sample_observations
from generalization_data import FIELD, FRESH

OUTPUT = RESULTS.parent / "knn_study"
HOLDOUT = OUTPUT / "fresh_manifest.csv"


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    if HOLDOUT.exists():
        rows = pd.read_csv(HOLDOUT)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for expected, actual in zip(rows.sha256, pool.map(download, rows.to_dict("records")), strict=True):
                if expected != actual["sha256"]:
                    raise ValueError(f"Photo changed after the audit: {actual['path']}")
        return

    previous = pd.concat([pd.read_csv(p) for p in (FIELD, FRESH, RESULTS / "external_manifest.csv")])
    blocked = set(previous.observer)

    def choose(item):
        label, taxon = item
        rows = sample_observations(label, taxon, "2020-01-01", "2021-12-31",
                                   OUTPUT / "photos" / "field", blocked, goal=120)
        print(f"{label}: {min(len(rows), 40)} eligible fresh photos", flush=True)
        return rows[:40]

    with ThreadPoolExecutor(max_workers=2) as pool:
        candidates = [row for batch in pool.map(choose, TAXA.items()) for row in batch]
    with ThreadPoolExecutor(max_workers=12) as pool:
        fresh = pd.DataFrame(pool.map(download, candidates))

    raw = sorted(path for path in RAW.rglob("*") if path.suffix.lower() in EXTENSIONS)
    known = raw + [ROOT / path for path in previous.path]
    paths = known + [ROOT / path for path in fresh.path]
    inventory, images, unreadable = scan_images(paths, lambda p: p.parent.name,
                                               lambda p: p.relative_to(ROOT).as_posix())
    if unreadable:
        raise ValueError(f"Could not audit {len(unreadable)} photos.")
    groups = create_groups(len(inventory), related_pairs(inventory, images))
    used = set(groups[:len(known)])
    keep = []
    for group in groups[len(known):]:
        keep.append(group not in used)
        used.add(group)
    retained = fresh[keep]
    if set(retained.label) != set(CLASSES):
        raise ValueError("Fresh test must cover all five classes.")
    fresh.drop(retained.index).to_csv(OUTPUT / "overlap_exclusions.csv", index=False)
    retained.to_csv(HOLDOUT, index=False)
    print(f"Fixed fresh sample: {retained.label.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    prepare()
