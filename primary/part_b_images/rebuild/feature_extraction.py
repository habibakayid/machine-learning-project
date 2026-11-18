"""Handcrafted descriptors shared by training and Streamlit."""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import hog, local_binary_pattern

from dataset_preparation import RESULTS, ROOT

SIZE = 128
CACHE = RESULTS.parent / "cache"
FEATURE_GROUPS = {
    "legacy": ("legacy_hog", "rgb", "legacy_hsv"),
    "crop": ("color", "hog", "lbp", "edges"),
    "pad": ("color", "hog", "lbp", "edges"),
    "local-pad": ("local",),
    "local-crop": ("local",),
}
Y, X = np.mgrid[:SIZE, :SIZE].astype(np.float32)
X = (X - (SIZE - 1) / 2) / (SIZE / 2)
Y = (Y - (SIZE - 1) / 2) / (SIZE / 2)
RADIUS = np.hypot(X, Y)
ANGLE = np.arctan2(Y, X)
CENTER = np.exp(-1.5 * RADIUS ** 2)


def prepared_pixels(image, mode, augment=None):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if augment == "flip":
        image = ImageOps.mirror(image)
    elif augment == "mild":
        # Crop the expanded rotation to avoid adding a colored border.
        image = image.rotate(9, Image.Resampling.BICUBIC, expand=False)
        w, h = image.size
        mx, my = min(round(h * .085), w // 4), min(round(w * .085), h // 4)
        image = image.crop((mx, my, w - mx, h - my))
        image = ImageEnhance.Brightness(image).enhance(1.07)
        image = ImageEnhance.Contrast(image).enhance(.92)

    method = Image.Resampling.LANCZOS
    if mode == "legacy":
        resized = ImageOps.pad(image, (64, 64), method, color="white")
        return np.asarray(resized, dtype=np.float32) / 255, None
    if mode == "crop":
        return np.asarray(ImageOps.fit(image, (SIZE, SIZE), method)), np.ones((SIZE, SIZE))
    resized = np.asarray(ImageOps.contain(image, (SIZE, SIZE), method))
    h, w = resized.shape[:2]
    top, left = (SIZE - h) // 2, (SIZE - w) // 2
    mask = np.zeros((SIZE, SIZE), dtype=np.float32)
    mask[top:top + h, left:left + w] = 1
    pixels = np.pad(resized, ((top, SIZE - h - top), (left, SIZE - w - left), (0, 0)),
                    mode="reflect")
    return pixels, mask


def histogram(codes, bins, weights=None, root=True):
    counts = np.bincount(codes.ravel(), weights=None if weights is None else weights.ravel(),
                         minlength=bins).astype(np.float32)
    counts /= max(counts.sum(), 1e-8)
    return np.sqrt(counts) if root else counts


def quantize(values, bins):
    return np.clip((values * bins).astype(np.int32), 0, bins - 1)


def legacy_features(rgb):
    def channels(image):
        return [histogram(quantize(ch, 16), 16, root=False) for ch in image.transpose(2, 0, 1)]

    hsv = rgb2hsv(rgb)
    quadrants = [hsv[y:y + 32, x:x + 32] for y in (0, 32) for x in (0, 32)]
    return {
        "legacy_hog": hog(rgb, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1),
        "rgb": np.concatenate(channels(rgb)),
        "legacy_hsv": np.concatenate([feature for region in [hsv, *quadrants]
                                       for feature in channels(region)]),
    }


def color_features(rgb, mask):
    h, s, v = rgb2hsv(rgb).transpose(2, 0, 1)
    light, a, b = rgb2lab(rgb).transpose(2, 0, 1)
    hue = quantize((h + 1 / 48) % 1, 24)
    sat, val = quantize(s, 5), quantize(v, 8)
    # The first hue bucket is reserved for nearly neutral pixels.
    hs = np.where(s < .11, 0, 1 + hue * 5 + sat)
    sv = sat * 8 + val
    ab = quantize((a + 105) / 210, 12) * 12 + quantize((b + 105) / 210, 12)
    hue_light = np.where(s < .14, 0, 1 + quantize(h, 12)) * 6 + quantize(light / 100, 6)
    green = (h > .20) & (h < .49) & (s > .22)
    quadrants = [(Y < 0) & (X < 0), (Y < 0) & (X >= 0),
                 (Y >= 0) & (X < 0), (Y >= 0) & (X >= 0)]
    regions = [mask, mask * CENTER, mask * CENTER ** 3,
               mask * (1 - CENTER), mask * CENTER * np.where(green, .16, 1)]
    features = [histogram(code, bins, region)
                for region in regions
                for code, bins in ((hs, 121), (sv, 40), (ab, 144), (hue_light, 78))]
    features.extend(histogram(hs, 121, mask * quadrant) for quadrant in quadrants)
    return np.concatenate(features)


def texture_features(gray, mask):
    gray8 = np.uint8(np.round(gray * 255))
    features = []
    for points, radius in ((8, 1), (16, 2), (24, 3)):
        codes = local_binary_pattern(gray8, points, radius, method="uniform").astype(np.int32)
        for region in (mask, mask * CENTER, mask * CENTER ** 3, mask * (1 - CENTER)):
            features.append(histogram(codes, points + 2, region))
    return np.concatenate(features)


def edge_features(gray, mask):
    features = []
    for sigma in (1, 3, 7):
        dy, dx = np.gradient(gaussian_filter(gray, sigma))
        magnitude = np.hypot(dx, dy)
        orientation = np.arctan2(dy, dx)
        angle = quantize((orientation % np.pi) / np.pi, 12)
        radial = quantize(((orientation - ANGLE) % np.pi) / np.pi, 12)
        strength = quantize(np.log1p(magnitude * 500) / 4, 12)
        for region in (mask, mask * CENTER, mask * CENTER ** 3, mask * (1 - CENTER)):
            features.extend((histogram(angle, 12, region * magnitude),
                             histogram(radial, 12, region * magnitude),
                             histogram(strength, 12, region)))
    return np.concatenate(features)


def extract(image, mode, augment=None):
    if mode.startswith("local-"):
        from local_features import describe_patches
        pixels, mask = prepared_pixels(image, mode.removeprefix("local-"), augment)
        return {"local": describe_patches(pixels.astype(np.float32) / 255, mask)}
    pixels, mask = prepared_pixels(image, mode, augment)
    if mode == "legacy":
        return legacy_features(pixels)
    rgb = pixels.astype(np.float32) / 255
    gray = rgb2gray(rgb)
    return {
        "color": color_features(rgb, mask),
        "hog": hog(rgb, pixels_per_cell=(16, 16), cells_per_block=(2, 2), channel_axis=-1),
        "lbp": texture_features(gray, mask),
        "edges": edge_features(gray, mask),
    }


def cached_features(rows, mode, augment=None, workers=5):
    CACHE.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes())
    if mode.startswith("local-"):
        fingerprint.update(Path(__file__).with_name("local_features.py").read_bytes())
    fingerprint.update(rows[["path", "sha256"]].to_csv(index=False).encode())
    key = fingerprint.hexdigest()[:16]
    cache_file = CACHE / f"{mode}-{augment or 'original'}-{key}.npz"
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def from_path(item):
        path, digest = item
        raw = (ROOT / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Image changed after audit: {path}")
        with Image.open(BytesIO(raw)) as image:
            return extract(image, mode, augment)

    parts = {name: [] for name in FEATURE_GROUPS[mode]}
    print(f"Extracting {mode}/{augment or 'original'} from {len(rows)} images", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, features in enumerate(pool.map(from_path, zip(rows.path, rows.sha256)), 1):
            for name in parts:
                parts[name].append(features[name])
            if index % 1000 == 0:
                print(f"  {index}/{len(rows)}", flush=True)
    arrays = {name: np.asarray(vectors, dtype=np.float32) for name, vectors in parts.items()}
    temp_file = cache_file.with_suffix(".tmp")
    with temp_file.open("wb") as output:
        np.savez(output, **arrays)
    temp_file.replace(cache_file)
    return arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=FEATURE_GROUPS, default=["legacy", "pad", "crop"])
    parser.add_argument("--workers", type=int, default=5)
    options = parser.parse_args()
    manifest = pd.read_csv(RESULTS / "manifest.csv")
    development = manifest[manifest.split.isin(["train", "validation"])]
    for mode in options.modes:
        for name, values in cached_features(development, mode, workers=options.workers).items():
            print(f"{mode}/{name}: {values.shape}", flush=True)
