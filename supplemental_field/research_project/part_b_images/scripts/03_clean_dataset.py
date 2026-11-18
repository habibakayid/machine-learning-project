
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv

# Paths
DATASET_PATH = Path("part_b_images/data/raw/flowers")
OUTPUT_PATH = Path("part_b_images/data/processed")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
extensions = {".jpg", ".jpeg", ".png"}

# Group images by their content hash
hash_groups = defaultdict(list)

for class_name in classes:
    class_path = DATASET_PATH / class_name

    for image_path in sorted(class_path.iterdir()):
        if image_path.suffix.lower() not in extensions:
            continue

        if not image_path.is_file():
            continue

        image_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        hash_groups[image_hash].append({
            "path": image_path.as_posix(),
            "label": class_name
        })

# Separate clean and excluded images
clean_images = []
excluded_images = []

for group in hash_groups.values():

    labels = {item["label"] for item in group}

    if len(labels) > 1:
        # Conflicting labels: exclude all copies
        for item in group:
            excluded_images.append({
                **item,
                "reason": "conflicting_labels"
            })

    else:
        # Keep one copy of each unique image
        clean_images.append(group[0])

        for item in group[1:]:
            excluded_images.append({
                **item,
                "reason": "same_class_duplicate"
            })

# Save clean dataset manifest
clean_images.sort(key=lambda x: x["path"])

with open(
    OUTPUT_PATH / "clean_images.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["path", "label"]
    )
    writer.writeheader()
    writer.writerows(clean_images)

# Save excluded image records
with open(
    OUTPUT_PATH / "excluded_images.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["path", "label", "reason"]
    )
    writer.writeheader()
    writer.writerows(excluded_images)

# Summary
class_counts = Counter(
    item["label"] for item in clean_images
)

print("\n===== DATA CLEANING REPORT =====")

print(f"Original Images: {sum(len(g) for g in hash_groups.values())}")
print(f"Excluded Images: {len(excluded_images)}")
print(f"Clean Images: {len(clean_images)}")

print("\nImages per Class:")

for class_name in classes:
    print(f"{class_name}: {class_counts[class_name]}")

print("\nCleaning completed successfully!")
