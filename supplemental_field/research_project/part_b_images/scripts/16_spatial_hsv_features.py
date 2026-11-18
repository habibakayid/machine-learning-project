
from pathlib import Path
import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from skimage.color import rgb2hsv

# =====================================
# 1. Paths and Configuration
# =====================================

SPLIT_PATH = Path(
    "part_b_images/data/processed/splits"
)

OUTPUT_PATH = Path(
    "part_b_images/data/processed/features_spatial"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
BINS = 16

# =====================================
# 2. Image Preprocessing
# =====================================

def load_image(image_path):

    with Image.open(image_path) as img:

        img = ImageOps.exif_transpose(img).convert("RGB")

        img = ImageOps.pad(
            img,
            IMAGE_SIZE,
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255)
        )

        image = np.asarray(
            img, dtype=np.float32
        ) / 255.0

    return image


# =====================================
# 3. HSV Histogram
# =====================================

def hsv_histogram(region):

    features = []

    for channel in range(3):

        hist, _ = np.histogram(
            region[:, :, channel],
            bins=BINS,
            range=(0.0, 1.0)
        )

        hist = hist.astype(np.float32)
        hist /= max(hist.sum(), 1.0)

        features.extend(hist)

    return np.array(features, dtype=np.float32)


# =====================================
# 4. Spatial Feature Extraction
# =====================================

def extract_spatial_features(image):

    hsv = rgb2hsv(image)

    # Global image histogram
    global_features = hsv_histogram(hsv)

    # Divide into 4 spatial regions
    regions = []

    for row in np.array_split(hsv, 2, axis=0):
        for region in np.array_split(row, 2, axis=1):
            regions.append(region)

    spatial_features = [
        hsv_histogram(region)
        for region in regions
    ]

    return np.concatenate(
        [global_features] + spatial_features
    )


# =====================================
# 5. Process Each Split
# =====================================

def process_split(split_name):

    df = pd.read_csv(
        SPLIT_PATH / f"{split_name}.csv"
    )

    X = np.empty(
        (len(df), 240),
        dtype=np.float32
    )

    y = df["label"].to_numpy(dtype="U16")

    for i, image_path in enumerate(df["path"]):

        image = load_image(image_path)

        X[i] = extract_spatial_features(image)

        if (i + 1) % 500 == 0:
            print(f"{split_name}: {i + 1} processed")

    np.savez_compressed(
        OUTPUT_PATH / f"{split_name}_spatial_hsv.npz",
        X=X,
        y=y
    )

    print(f"\n{split_name} completed!")
    print("Feature Shape:", X.shape)
    print("Label Shape:", y.shape)



# =====================================
# 6. Main
# =====================================

if __name__ == "__main__":

    process_split("train")
    process_split("validation")
    process_split("test")

    print("\nSpatial HSV Feature Extraction Completed!")
