"""Import a locally obtained Kaggle collection and verify frozen byte hashes."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-folder", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--research", action="store_true", help="Also restore the historical research tree's Kaggle photos.")
    args = parser.parse_args()
    inventory = pd.read_csv(ROOT / "primary/part_b_images/rebuild/results/raw_inventory.csv")
    archive = zipfile.ZipFile(args.archive) if args.archive else None
    entries = archive.namelist() if archive else [p for p in args.from_folder.rglob("*") if p.is_file()]
    lookup = {}
    for entry in entries:
        path = Path(entry)
        if path.parent.name in set(inventory.label):
            lookup.setdefault((path.parent.name, path.name), []).append(entry)
    targets = [ROOT / "primary"]
    if args.research:
        targets.append(ROOT / "supplemental_field/research_project")
    try:
        for row in inventory.itertuples():
            matches = lookup.get((row.label, Path(row.path).name), [])
            data = None
            for match in matches:
                candidate = archive.read(match) if archive else match.read_bytes()
                if hashlib.sha256(candidate).hexdigest() == row.sha256:
                    data = candidate
                    break
            if data is None:
                raise ValueError(f"Missing or changed original image: {row.path}")
            with Image.open(BytesIO(data)) as image:
                image.verify()
            for target in targets:
                destination = target / row.path
                if destination.exists() and destination.read_bytes() != data:
                    raise ValueError(f"Refusing to overwrite a different file: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(data)
        samples = json.loads((ROOT / "app/samples/MANIFEST.json").read_text())
        for sample in samples:
            destination = ROOT / "app" / sample["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "primary" / sample["original_dataset_path"], destination)
        report = {"images_verified": len(inventory), "classes": inventory.label.value_counts().to_dict(),
                  "hash_and_image_validation": "passed", "research_tree_restored": args.research}
        (ROOT / "verification/DATASET_IMPORT.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        if archive:
            archive.close()


if __name__ == "__main__":
    main()
