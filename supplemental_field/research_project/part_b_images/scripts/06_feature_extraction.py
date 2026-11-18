
from pathlib import Path
import numpy as np
from skimage.feature import hog

# =====================================
# Paths
# =====================================

INPUT_PATH = Path(
    "part_b_images/data/processed/arrays"
)

OUTPUT_PATH = Path(
    "part_b_images/data/processed/features"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =====================================
# HOG Configuration
# =====================================

HOG_SETTINGS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "channel_axis": -1,
    "feature_vector": True
}

# =====================================
# Extract Features from One Image
# =====================================

def extract_hog(image):

    features = hog(
        image,
        **HOG_SETTINGS
    )

    return features.astype(np.float32)


# =====================================
# Process Dataset Split
# =====================================

def process_split(split_name):

    input_file = INPUT_PATH / f"{split_name}.npz"

    with np.load(input_file) as data:
        images = data["X"]
        labels = data["y"]

    print(f"\nProcessing {split_name}...")

    if len(images) == 0:
        raise ValueError("Dataset split is empty!")

    # Extract first image features
    first_features = extract_hog(images[0])

    number_of_features = len(first_features)

    # Allocate features array
    X_features = np.empty(
        (len(images), number_of_features),
        dtype=np.float32
    )

    X_features[0] = first_features

    # Process remaining images
    for i in range(1, len(images)):

        X_features[i] = extract_hog(images[i])

        if (i + 1) % 500 == 0:
            print(f"{i + 1} images processed")

    # Save features and labels
    np.savez_compressed(
        OUTPUT_PATH / f"{split_name}_hog.npz",
        X=X_features,
        y=labels
    )

    print(f"{split_name} completed!")
    print(f"Features Shape: {X_features.shape}")
    print(f"Labels Shape: {labels.shape}")

    return number_of_features


# =====================================
# Main
# =====================================

if __name__ == "__main__":

    train_features = process_split("train")
    val_features = process_split("validation")
    test_features = process_split("test")

    assert train_features == val_features == test_features

    print("\n===== FEATURE EXTRACTION REPORT =====")
    print("Method: HOG")
    print(f"Features per Image: {train_features}")
    print("Feature Extraction completed successfully!")
