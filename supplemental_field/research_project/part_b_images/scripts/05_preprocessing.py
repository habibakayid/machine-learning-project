
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# =====================================
# Configuration
# =====================================

SPLIT_PATH = Path("part_b_images/data/processed/splits")
OUTPUT_PATH = Path("part_b_images/data/processed/arrays_96")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (96, 96)

# =====================================
# Preprocess One Image
# =====================================

def preprocess_image(image_path):

    with Image.open(image_path) as img:

        # Correct orientation and convert to RGB
        img = ImageOps.exif_transpose(img).convert("RGB")

        # Resize while preserving aspect ratio
        img = ImageOps.pad(
            img,
            IMAGE_SIZE,
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255)
        )

        # Convert image to NumPy array
        image_array = np.asarray(img, dtype=np.float32)

        # Normalize pixel values to [0, 1]
        image_array = image_array / 255.0

    return image_array


# =====================================
# Process Each Dataset Split
# =====================================

def process_split(split_name, csv_file):

    df = pd.read_csv(SPLIT_PATH / csv_file)

    images = np.empty(
    (len(df), 96, 96, 3),
    dtype=np.float32
)


    labels = df["label"].to_numpy(dtype="U16")

    for index, image_path in enumerate(df["path"]):

        images[index] = preprocess_image(image_path)

        if (index + 1) % 500 == 0:
            print(f"{split_name}: {index + 1} images processed")

    # Save images and labels
    np.savez_compressed(
        OUTPUT_PATH / f"{split_name}.npz",
        X=images,
        y=labels
    )

    print(f"\n{split_name} completed!")
    print(f"Images Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")
    print(f"Min Pixel Value: {images.min():.4f}")
    print(f"Max Pixel Value: {images.max():.4f}")

    return len(images)


# =====================================
# Main
# =====================================

if __name__ == "__main__":

    train_count = process_split("train", "train.csv")

    val_count = process_split("validation", "validation.csv")

    test_count = process_split("test", "test.csv")

    print("\n===== PREPROCESSING REPORT =====")

    print(f"Training Images: {train_count}")
    print(f"Validation Images: {val_count}")
    print(f"Testing Images: {test_count}")

    print(f"Final Image Size: {IMAGE_SIZE}")
    print("Pixel Values: 0 to 1")

    print("\nPreprocessing completed successfully!")
