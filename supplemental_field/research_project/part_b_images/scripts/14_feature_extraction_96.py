
from pathlib import Path
import numpy as np
from skimage.feature import hog

# =====================================
# 1. Paths
# =====================================

INPUT_PATH = Path(
    "part_b_images/data/processed/arrays_96"
)

OUTPUT_PATH = Path(
    "part_b_images/data/processed/features_96"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. Configuration
# =====================================

IMAGE_SIZE = (96, 96)
COLOR_BINS = 16

HOG_SETTINGS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "channel_axis": -1,
    "feature_vector": True
}

# =====================================
# 3. HOG Feature Extraction
# =====================================

def extract_hog(image):

    features = hog(
        image,
        **HOG_SETTINGS
    )

    return features.astype(np.float32)


# =====================================
# 4. Color Histogram Extraction
# =====================================

def extract_color(image):

    histograms = []

    for channel in range(3):

        histogram, _ = np.histogram(
            image[:, :, channel],
            bins=COLOR_BINS,
            range=(0.0, 1.0)
        )

        histogram = histogram.astype(np.float32)

        histogram /= max(histogram.sum(), 1)

        histograms.append(histogram)

    return np.concatenate(histograms)


# =====================================
# 5. Process Dataset Split
# =====================================

def process_split(split_name):

    print(f"\nProcessing {split_name}...")

    input_file = INPUT_PATH / f"{split_name}.npz"

    with np.load(input_file, allow_pickle=False) as data:
        images = data["X"]
        labels = data["y"]

    if images.shape[1:] != (96, 96, 3):
        raise ValueError(
            f"Expected 96x96 RGB images, got {images.shape}"
        )

    # Extract one sample to determine feature dimensions
    hog_size = len(extract_hog(images[0]))
    color_size = len(extract_color(images[0]))

    # Allocate arrays
    X_hog = np.empty(
        (len(images), hog_size),
        dtype=np.float32
    )

    X_color = np.empty(
        (len(images), color_size),
        dtype=np.float32
    )

    # Extract features
    for i, image in enumerate(images):

        X_hog[i] = extract_hog(image)
        X_color[i] = extract_color(image)

        if (i + 1) % 500 == 0:
            print(f"{i + 1} images processed")

    # =====================================
    # 6. Save Features
    # =====================================

    np.savez_compressed(
        OUTPUT_PATH / f"{split_name}_hog.npz",
        X=X_hog,
        y=labels
    )

    np.savez_compressed(
        OUTPUT_PATH / f"{split_name}_color.npz",
        X=X_color,
        y=labels
    )

    print(f"\n{split_name} completed!")
    print(f"HOG Shape: {X_hog.shape}")
    print(f"Color Shape: {X_color.shape}")
    print(f"Combined Features: {hog_size + color_size}")

    assert hog_size == 4356
    assert color_size == 48

    return hog_size, color_size


# =====================================
# 7. Main
# =====================================

if __name__ == "__main__":

    train_sizes = process_split("train")
    val_sizes = process_split("validation")

    assert train_sizes == val_sizes

    print("\n===== 96x96 FEATURE EXTRACTION REPORT =====")
    print(f"HOG Features: {train_sizes[0]}")
    print(f"Color Features: {train_sizes[1]}")
    print(f"Total Features: {sum(train_sizes)}")
    print("\nFeature Extraction Completed!")
