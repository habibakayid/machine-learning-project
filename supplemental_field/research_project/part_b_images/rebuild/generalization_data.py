"""Freeze disjoint iNaturalist development and fresh test manifests."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from dataset_preparation import CLASSES, RAW, ROOT, RESULTS, related_pairs, create_groups, scan_images
from external_dataset import TAXA, download, sample_observations

OUTPUT = RESULTS.parent / "generalization"
FRESH = OUTPUT / "fresh_manifest.csv"
FIELD = OUTPUT / "field_manifest.csv"


def collect(name, start, end, cap, blocked):
    folder = OUTPUT / "photos" / name

    def collect_class(item):
        label, taxon = item
        rows = sample_observations(label, taxon, start, end, folder, blocked,
                                   page_limit=6, goal=cap * 2)[:cap]
        print(f"{name}: {len(rows)} eligible {label} photos selected", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = list(pool.map(collect_class, TAXA.items()))
    rows = [row for batch in batches for row in batch]
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = pd.DataFrame(pool.map(download, rows))
    print(f"{name} downloaded: {rows.label.value_counts().to_dict()}", flush=True)
    return rows


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    if FIELD.exists() and FRESH.exists():
        for manifest in (FIELD, FRESH):
            rows = pd.read_csv(manifest)
            with ThreadPoolExecutor(max_workers=8) as pool:
                for expected, actual in zip(rows.sha256,
                        pool.map(download, rows.to_dict("records")), strict=True):
                    if expected != actual["sha256"]:
                        raise ValueError(f"Photo hash changed for {actual['path']}")
        return

    old = pd.read_csv(RESULTS / "external_manifest.csv")
    fresh = collect("fresh_2026", "2026-01-01", "2026-10-31", 38, set(old.observer))
    field = collect("field_2022_2024", "2022-01-01", "2024-12-31", 120,
                    set(old.observer) | set(fresh.observer))
    # Check the original raw collection and the earlier external test, then
    # reserve fresh images before allowing any matches into development.
    raw = sorted(p for p in RAW.rglob("*") if p.is_file()
                 and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    paths = raw + [ROOT / p for p in pd.concat([old, fresh, field]).path]
    inventory, fingerprints, unreadable = scan_images(
        paths, lambda p: p.parent.name, lambda p: p.relative_to(ROOT).as_posix()
    )
    if unreadable:
        raise ValueError(f"Unable to audit {len(unreadable)} images.")
    groups = create_groups(len(inventory), related_pairs(inventory, fingerprints))
    seen = set(groups[:len(raw) + len(old)])
    exclusions = []
    offset = len(raw) + len(old)
    for rows, destination in ((fresh, FRESH), (field, FIELD)):
        keep = []
        for i, group in enumerate(groups[offset:offset + len(rows)]):
            if group in seen:
                exclusions.append(rows.iloc[i].to_dict())
            else:
                seen.add(group)
                keep.append(i)
        offset += len(rows)
        retained = rows.iloc[keep]
        if set(retained.label) != set(CLASSES):
            raise ValueError(f"Missing classes from {destination.name}")
        retained.to_csv(destination, index=False)
        print(f"{destination.name}: {retained.label.value_counts().to_dict()}", flush=True)
    pd.DataFrame(exclusions, columns=fresh.columns).to_csv(OUTPUT / "overlap_exclusions.csv",
                                                         index=False)


if __name__ == "__main__":
    prepare()
