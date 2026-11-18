"""Small, interpretable color and shape descriptors for the generalization study."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import local_binary_pattern

from dataset_preparation import ROOT
from feature_extraction import CENTER, X, Y, histogram, prepared_pixels, quantize


def foreground(rgb, valid):
    """Soft emphasis on colors that differ from the border, with a green penalty."""
    hsv = rgb2hsv(rgb)
    h, s, v = hsv.transpose(2, 0, 1)
    lab = rgb2lab(rgb)
    ys, xs = np.where(valid > 0)
    border = (ys - ys.min() < 5) | (ys.max() - ys < 5) | (
        xs - xs.min() < 5) | (xs.max() - xs < 5)
    reference = np.median(lab[ys[border], xs[border]], axis=0)
    contrast = np.clip(np.linalg.norm(lab - reference, axis=2) / 55, 0, 1)
    green = (h > .21) & (h < .48) & (s > .23)
    # A positive floor keeps shadows, pale petals, and off-center flowers visible.
    weights = (.20 + .80 * contrast) * np.where(green, .23, 1.)
    weights *= .50 + .50 * np.clip(v / .45, 0, 1)
    weights *= .70 + .30 * CENTER
    return hsv, lab, valid * weights


def color_summary(hsv, lab, weights):
    h, s, v = hsv.transpose(2, 0, 1)
    light, a, b = lab.transpose(2, 0, 1)
    hue = np.where(s < .13, 0, 1 + quantize((h + 1 / 32) % 1, 16))
    hs = hue * 4 + quantize(s, 4)
    sv = quantize(s, 5) * 7 + quantize(v, 7)
    ab = quantize((a + 100) / 200, 9) * 9 + quantize((b + 100) / 200, 9)
    hl = hue * 5 + quantize(light / 100, 5)
    return np.concatenate([histogram(code, bins, weights)
                           for code, bins in ((hs, 68), (sv, 35), (ab, 81), (hl, 85))])


def shape_summary(gray, weights):
    cx = (X * weights).sum() / weights.sum()
    cy = (Y * weights).sum() / weights.sum()
    radial = np.arctan2(Y - cy, X - cx)
    result = []
    for sigma in (1, 3, 6):
        dy, dx = np.gradient(gaussian_filter(gray, sigma))
        angle = np.arctan2(dy, dx)
        magnitude = np.hypot(dx, dy)
        result.extend([
            histogram(quantize((angle % np.pi) / np.pi, 8), 8, weights * magnitude),
            histogram(quantize(((angle - radial) % np.pi) / np.pi, 8), 8, weights * magnitude),
            histogram(quantize(np.log1p(magnitude * 500) / 4, 8), 8, weights),
        ])
    gray8 = (gray * 255).round().astype(np.uint8)
    for points, radius in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray8, points, radius, method="uniform").astype(np.int32)
        result.append(histogram(lbp, points + 2, weights))
    return np.concatenate(result)


def extract_compact(image, mode):
    pixels, valid = prepared_pixels(image, mode.removeprefix("compact-"))
    rgb = pixels.astype(np.float32) / 255
    hsv, lab, emphasis = foreground(rgb, valid)
    gray = rgb2gray(rgb)
    return {
        "global_color": color_summary(hsv, lab, valid),
        "foreground_color": color_summary(hsv, lab, emphasis),
        "global_shape": shape_summary(gray, valid),
        "foreground_shape": shape_summary(gray, emphasis),
    }


def compact_cache(rows, mode, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes())
    fingerprint.update(Path(__file__).with_name("feature_extraction.py").read_bytes())
    fingerprint.update(rows[["path", "sha256"]].to_csv(index=False).encode())
    path = directory / f"{mode}-{fingerprint.hexdigest()[:16]}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return dict(saved)

    def read(row):
        file, expected = row
        raw = (ROOT / file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Photo changed after the audit: {file}")
        with Image.open(BytesIO(raw)) as image:
            return extract_compact(image, mode)

    print(f"Extracting {mode} from {len(rows)} images.", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        features = list(pool.map(read, zip(rows.path, rows.sha256)))
    arrays = {group: np.asarray([row[group] for row in features], dtype=np.float32)
              for group in features[0]}
    np.savez(path, **arrays)
    return arrays
