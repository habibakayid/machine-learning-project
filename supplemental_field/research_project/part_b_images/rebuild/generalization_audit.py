"""Compare framing and color cues in Kaggle development and the published field test."""

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from dataset_preparation import RESULTS, ROOT, SEED
from feature_extraction import prepared_pixels
from generalization_data import OUTPUT
from generalization_features import foreground


def inspect(row):
    with Image.open(ROOT / row.path) as img:
        image = ImageOps.exif_transpose(img)
        pixels, valid = prepared_pixels(image, "pad")
        hsv, _, cue = foreground(pixels.astype(np.float32) / 255, valid)
        w, h = image.size
    ys, xs = np.where(valid > 0)
    side = min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    y0 = (ys.max() + ys.min() + 1 - side) // 2
    x0 = (xs.max() + xs.min() + 1 - side) // 2
    hue, sat, _ = hsv.transpose(2, 0, 1)
    green = (hue > .21) & (hue < .48) & (sat > .23)
    return {"path": row.path, "label": row.label, "source": row.source,
            "area_removed": 1 - min(w, h) / max(w, h),
            "cue_removed": 1 - cue[y0:y0 + side, x0:x0 + side].sum() / cue.sum(),
            "green_fraction": (green * valid).sum() / valid.sum()}


def contact_sheet(kaggle, field, label):
    subsets = [frame[frame.label == label].sample(8, random_state=SEED)
               for frame in (kaggle, field)]
    fig, axes = plt.subplots(8, 4, figsize=(9, 15))
    for i, pair in enumerate(zip(*(subset.itertuples(index=False) for subset in subsets))):
        for j, row in enumerate(pair):
            with Image.open(ROOT / row.path) as img:
                original = ImageOps.exif_transpose(img).convert("RGB")
                full = ImageOps.contain(original, (216, 155))
                crop = ImageOps.fit(original, (155, 155))
            for col, view in enumerate((full, crop)):
                ax = axes[i, j * 2 + col]
                ax.imshow(view)
                ax.axis("off")
            axes[i, j * 2].set_title(row.taxon if j else row.path.rsplit("/", 1)[-1],
                                     fontsize=8)
    for ax, title in zip(axes[0], ("Kaggle full", "Kaggle crop", "iNat full", "iNat crop")):
        ax.text(.5, 1.25, title, ha="center", transform=ax.transAxes, weight="bold")
    fig.suptitle(label.title() + " photos, seeded sample", y=.99)
    fig.tight_layout(pad=.8)
    fig.savefig(OUTPUT / f"{label}_framing.png", dpi=120)
    plt.close(fig)


def audit():
    OUTPUT.mkdir(exist_ok=True)
    manifest = pd.read_csv(RESULTS / "manifest.csv")
    kaggle = manifest[manifest.split.isin(["train", "validation"])].assign(source="Kaggle")
    field = pd.read_csv(RESULTS / "external_manifest.csv").assign(source="2025 iNat")
    sample = kaggle.groupby("label", group_keys=False).sample(100, random_state=SEED)
    rows = pd.concat([sample, field], ignore_index=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        details = pd.DataFrame(pool.map(inspect, rows.itertuples(index=False)))
    details.to_csv(OUTPUT / "framing_details.csv", index=False)
    summary = details.groupby(["source", "label"])[
        ["area_removed", "cue_removed", "green_fraction"]
    ].median()
    summary.to_csv(OUTPUT / "framing_summary.csv")
    field.groupby(["label", "taxon"]).size().rename("photos").to_csv(OUTPUT / "older_field_taxa.csv")
    for label in ("rose", "tulip"):
        contact_sheet(kaggle, field, label)
    print(summary, flush=True)


if __name__ == "__main__":
    audit()
