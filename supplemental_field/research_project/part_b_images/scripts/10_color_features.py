
from pathlib import Path
import numpy as np

# =====================================
# Paths and Configuration
# =====================================

INPUT_PATH = Path(
    "part_b_images/data/processed/arrays"
)

OUTPUT_PATH = Path(
    "part_b_images/data/processed/features"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

BINS = 16

# =====================================
# Extract Color Histogram
# =====================================

def extract_color_features(image):

    histograms = []

    # Process Red, Green and Blue channels
    for channel in range(3):

        histogram, _ = np.histogram(
            image[:, :, channel],
            bins=BINS,
            range=(0.0, 1.0)
        )

        histogram = histogram.astype(np.float32)

        # Normalize histogram
        histogram /= max(histogram.sum(), 1)

        histograms.append(histogram)

    return np.concatenate(histograms)


# =====================================
# Process Dataset Split
# =====================================

def process_split(split_name):

    with np.load(
        INPUT_PATH / f"{split_name}.npz"
    ) as data:

        images = data["X"]
        labels = data["y"]

    features = np.array(
        [extract_color_features(img) for img in images],
        dtype=np.float32
    )

    np.savez_compressed(
        OUTPUT_PATH / f"{split_name}_color.npz",
        X=features,
        y=labels
    )

    print(f"\n{split_name} completed!")
    print("Features Shape:", features.shape)
    print("Labels Shape:", labels.shape)



# =====================================
# Main
# =====================================

if __name__ == "__main__":

    process_split("train")
    process_split("validation")
    process_split("test")

    print("\nColor Feature Extraction Completed!")
