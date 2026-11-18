from pathlib import Path
import matplotlib.pyplot as plt

import random
from PIL import Image


# Paths
DATASET_PATH = Path("part_b_images/data/raw/flowers")
FIGURES_PATH = Path("part_b_images/figures")

FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# Classes
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
image_extensions = {".jpg", ".jpeg", ".png"}

# Count images per class
image_counts = []

for class_name in classes:
    class_path = DATASET_PATH / class_name

    images = [
        file for file in class_path.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    image_counts.append(len(images))

# Create bar chart
plt.figure(figsize=(10, 6))

bars = plt.bar(classes, image_counts, color="steelblue")

plt.title("Flowers Dataset - Class Distribution")
plt.xlabel("Flower Class")
plt.ylabel("Number of Images")

# Show values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 10,
        str(int(height)),
        ha="center"
    )

plt.ylim(0, max(image_counts) * 1.15)
plt.tight_layout()

# Save chart
plt.savefig(FIGURES_PATH / "class_distribution.png", dpi=300)

plt.show()


# =====================================
# Visualizing Random Sample Images
# =====================================

random.seed(42)

fig, axes = plt.subplots(5, 3, figsize=(12, 16))

for i, class_name in enumerate(classes):

    class_path = DATASET_PATH / class_name

    image_files = [
        file for file in class_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in image_extensions
    ]

    selected_images = random.sample(image_files, 3)

    for j, image_path in enumerate(selected_images):

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

                axes[i, j].imshow(image)

            axes[i, j].set_title(class_name.capitalize())
            axes[i, j].axis("off")

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            axes[i, j].axis("off")

plt.suptitle("Random Samples from Flowers Dataset", fontsize=16)

plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "sample_images.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================
# Image Dimensions Analysis
# =====================================

import numpy as np

widths = []
heights = []
failed_images = []

for class_name in classes:
    class_path = DATASET_PATH / class_name

    for image_path in class_path.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in image_extensions:
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size

                widths.append(width)
                heights.append(height)

        except (OSError, ValueError) as e:
            failed_images.append(str(image_path))
            print(f"Error reading {image_path}: {e}")

# Display statistics
print("\n===== IMAGE DIMENSIONS =====")

print(f"Images analyzed: {len(widths)}")
print(f"Images failed: {len(failed_images)}")

if not widths:
    raise RuntimeError("No valid image dimensions found.")

print(f"Minimum Width: {min(widths)}")
print(f"Maximum Width: {max(widths)}")
print(f"Average Width: {np.mean(widths):.2f}")

print(f"Minimum Height: {min(heights)}")
print(f"Maximum Height: {max(heights)}")
print(f"Average Height: {np.mean(heights):.2f}")

# Plot dimensions distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(widths, bins=30, edgecolor="black")
axes[0].set_title("Image Width Distribution")
axes[0].set_xlabel("Width (Pixels)")
axes[0].set_ylabel("Number of Images")

axes[1].hist(heights, bins=30, edgecolor="black")
axes[1].set_title("Image Height Distribution")
axes[1].set_xlabel("Height (Pixels)")
axes[1].set_ylabel("Number of Images")

plt.suptitle("Flowers Dataset - Image Dimensions")

plt.tight_layout()

plt.savefig(
    FIGURES_PATH / "image_dimensions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================
# Image Quality Inspection
# =====================================

import hashlib
from collections import Counter

corrupted_images = []
duplicate_images = []
seen_hashes = {}
image_formats = Counter()

total_checked = 0

for class_name in classes:

    class_path = DATASET_PATH / class_name

    for image_path in class_path.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in image_extensions:
            continue

        total_checked += 1

        # Check if image is valid
        try:
            with Image.open(image_path) as img:
                image_formats[img.format] += 1
                img.load()

        except Exception as e:
            corrupted_images.append(str(image_path))
            print(f"Corrupted image: {image_path}")
            continue

        # Generate a unique hash for the image file
        with open(image_path, "rb") as file:
            image_hash = hashlib.sha256(file.read()).hexdigest()

        # Check for exact duplicates
        if image_hash in seen_hashes:
            duplicate_images.append(
                (str(image_path), seen_hashes[image_hash])
            )
        else:
            seen_hashes[image_hash] = str(image_path)


# Display Results
print("\n===== IMAGE QUALITY REPORT =====")

print(f"Total Images Checked: {total_checked}")
print(f"Corrupted Images: {len(corrupted_images)}")
print(f"Exact Duplicate Files: {len(duplicate_images)}")

print("\nImage Formats:")

for image_format, count in image_formats.items():
    print(f"{image_format}: {count}")

print("\nDuplicate Examples:")

for duplicate, original in duplicate_images[:5]:
    print(f"Duplicate: {duplicate}")
    print(f"Original: {original}")
    print("-" * 30)


# =====================================
# Detailed Duplicate Analysis
# =====================================

print("\n===== ALL DUPLICATE IMAGES =====")

same_class_duplicates = 0
cross_class_duplicates = 0

for index, (duplicate, original) in enumerate(duplicate_images, 1):

    duplicate_class = Path(duplicate).parent.name
    original_class = Path(original).parent.name

    if duplicate_class == original_class:
        duplicate_type = "SAME CLASS"
        same_class_duplicates += 1
    else:
        duplicate_type = "CROSS CLASS"
        cross_class_duplicates += 1

    print(f"\nDuplicate #{index}")
    print(f"Type: {duplicate_type}")
    print(f"Original: {original}")
    print(f"Duplicate: {duplicate}")

print("\n===== DUPLICATE SUMMARY =====")
print(f"Same-Class Duplicates: {same_class_duplicates}")
print(f"Cross-Class Duplicates: {cross_class_duplicates}")
print(f"Total Duplicates: {len(duplicate_images)}")
