"""Audit the raw images and create a reproducible, group-aware split."""

import hashlib
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE.parent / "data" / "raw" / "flowers"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CLASSES = ("daisy", "dandelion", "rose", "sunflower", "tulip")
SPLITS = ("train", "validation", "test")
SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fingerprints(image):
    gray = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    rgb = np.asarray(image.resize((12, 12), Image.Resampling.BILINEAR), dtype=np.int16)
    low = dctn(gray, norm="ortho")[:8, :8].ravel()[1:]
    return low > np.median(low), rgb


def scan_images(paths, label_for, path_for):
    rows, images, unreadable = [], [], []
    for path in paths:
        label = label_for(path)
        relative = path_for(path)
        try:
            data = path.read_bytes()
            with Image.open(BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            unreadable.append({"path": relative, "label": label, "reason": str(error)})
            continue
        photo_id = re.match(r"^(\d{6,})_[a-f0-9]{8,}", path.stem)
        rows.append({
            "path": relative, "label": label, "width": image.width, "height": image.height,
            "sha256": hashlib.sha256(data).hexdigest(),
            "pixel_sha256": hashlib.sha256(
                f"{image.width}x{image.height}".encode() + image.tobytes()
            ).hexdigest(),
            "photo_id": photo_id[1] if photo_id else "",
        })
        images.append((fingerprints(image), fingerprints(ImageOps.mirror(image))))
    return pd.DataFrame(rows), images, unreadable


def related_pairs(rows, images):
    """Find exact copies, shared Flickr IDs, and likely resized/flipped copies."""
    for field in ("sha256", "pixel_sha256", "photo_id"):
        seen = {}
        for i, key in enumerate(rows[field]):
            if key:
                if key in seen:
                    yield seen[key], i, field
                else:
                    seen[key] = i

    hashes = np.array([original[0] for original, _ in images])
    thumbnails = np.array([original[1] for original, _ in images])
    for i, (original, mirror) in enumerate(images):
        for fingerprint, thumbnail in (original, mirror):
            distances = np.count_nonzero(hashes[:i] != fingerprint, axis=1)
            candidates = np.flatnonzero(distances <= 9)
            differences = np.abs(thumbnails[candidates] - thumbnail).mean(axis=(1, 2, 3))
            for j in candidates[differences <= 22]:
                yield int(j), i, "perceptual"


def create_groups(size, pairs):
    parents = list(range(size))

    def root(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for first, second, _ in pairs:
        a, b = root(first), root(second)
        parents[max(a, b)] = min(a, b)
    return [root(i) for i in range(size)]


def prepare_dataset():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    folders = {path.name for path in RAW.iterdir() if path.is_dir()}
    if folders != set(CLASSES):
        raise ValueError(f"Unexpected class folders: {sorted(folders)}")
    paths = sorted(path for path in RAW.rglob("*") if path.suffix.lower() in EXTENSIONS)
    rows, images, excluded = scan_images(
        paths, lambda p: p.relative_to(RAW).parts[0], lambda p: p.relative_to(ROOT).as_posix()
    )
    rows.to_csv(RESULTS / "raw_inventory.csv", index=False)
    exact = defaultdict(list)
    for i, key in enumerate(rows["pixel_sha256"]):
        exact[key].append(i)
    keep = set(rows.index)
    for duplicates in exact.values():
        if len(duplicates) < 2:
            continue
        conflict = rows.loc[duplicates, "label"].nunique() > 1
        for i in duplicates if conflict else duplicates[1:]:
            keep.remove(i)
            excluded.append({
                "path": rows.at[i, "path"], "label": rows.at[i, "label"],
                "reason": "conflicting exact labels" if conflict else "same-class exact copy",
            })

    pairs = sorted(set(related_pairs(rows, images)))
    pd.DataFrame([
        (rows.at[a, "path"], rows.at[b, "path"], kind, rows.at[a, "label"] != rows.at[b, "label"])
        for a, b, kind in pairs
    ], columns=["first", "second", "match", "label_disagreement"]).to_csv(
        RESULTS / "related_photos.csv", index=False
    )
    pd.DataFrame(excluded, columns=["path", "label", "reason"]).to_csv(
        RESULTS / "excluded_images.csv", index=False
    )
    rows["group"] = create_groups(len(rows), pairs)
    rows = rows.loc[sorted(keep)].reset_index(drop=True)

    # Twenty group-stratified folds give 14 + 3 + 3 folds for the 70/15/15 split.
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    for fold, (_, held) in enumerate(splitter.split(rows, rows.label, rows.group)):
        rows.loc[held, "split"] = SPLITS[0 if fold < 14 else 1 if fold < 17 else 2]
    if rows.groupby("group").split.nunique().max() != 1:
        raise ValueError("Related photos crossed a partition.")
    counts = pd.crosstab(rows.label, rows.split).reindex(index=CLASSES, columns=SPLITS)
    if not (counts > 0).all().all():
        raise ValueError("Every partition must contain all five labels.")
    rows.to_csv(RESULTS / "manifest.csv", index=False)
    counts["raw"] = pd.Series([sum(p.relative_to(RAW).parts[0] == c for p in paths)
                               for c in CLASSES], index=CLASSES)
    counts.to_csv(RESULTS / "class_distribution.csv")
    sizes = rows.groupby("label")[["width", "height"]].agg(["min", "median", "max"])
    sizes.to_csv(RESULTS / "image_sizes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts[list(SPLITS)].plot.bar(stacked=True, ax=axes[0], rot=20, title="Images per class")
    axes[1].scatter(rows.width, rows.height, s=6, alpha=.2)
    axes[1].set(xlabel="Width in pixels", ylabel="Height in pixels", title="Source resolutions")
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset.png", dpi=150)
    plt.close(fig)
    print(counts, flush=True)
    print(f"{len(paths)} raw, {len(excluded)} excluded, {len(rows)} usable, "
          f"{rows.group.nunique()} groups, {len(pairs)} match records", flush=True)


if __name__ == "__main__":
    prepare_dataset()
