from pathlib import Path

# Dataset location
DATASET_PATH = Path("part_b_images/data/raw/flowers")

# Flower classes
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Supported image extensions
image_extensions = {".jpg", ".jpeg", ".png"}

total_images = 0

for class_name in classes:
    class_path = DATASET_PATH / class_name

    if not class_path.is_dir():
        print(f"Missing class folder: {class_name}")
        continue

    image_files = [
        file for file in class_path.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    ]


    count = len(image_files)
    total_images += count

    print(f"{class_name}: {count} images")

print("-" * 30)
print(f"Total Images: {total_images}")
print(f"Number of Classes: {len(classes)}")
