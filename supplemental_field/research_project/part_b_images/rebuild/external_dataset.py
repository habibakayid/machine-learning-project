"""Prepare a dated iNaturalist holdout without running any model predictions."""

import gzip
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image

from dataset_preparation import (
    EXTENSIONS, RAW, RESULTS, ROOT, SEED, create_groups, related_pairs, scan_images,
)

EXTERNAL = RESULTS.parent / "external_photos"
TAXA = {
    "daisy": 55563,       # Bellis perennis
    "dandelion": 47603,   # Taraxacum
    "rose": 53438,       # Rosa
    "sunflower": 57983,  # Helianthus annuus
    "tulip": 55800,      # Tulipa
}
LICENSES = {"cc0", "cc-by", "cc-by-sa"}
MAX_PER_CLASS = 50


def fetch(url):
    request = Request(url, headers={"User-Agent": "UniversityFlowersHoldout/1.0",
                                    "Accept-Encoding": "gzip"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=25) as response:
                data = response.read()
                return gzip.decompress(data) if response.headers.get("Content-Encoding") == "gzip" else data
        except (OSError, HTTPException):
            if attempt == 4:
                raise
            time.sleep(1 + attempt)


def sample_observations(class_name, taxon_id, start="2025-01-01", end="2025-12-31",
                        folder=EXTERNAL, excluded_observers=(), page_limit=6, goal=None):
    filters = {
        "taxon_id": taxon_id, "quality_grade": "research", "photo_license": "cc0,cc-by,cc-by-sa",
        "term_id": 12, "term_value_id": 13, "d1": start, "d2": end,
        "order_by": "id", "order": "asc", "per_page": 200,
    }
    records = []
    for page in range(1, page_limit + 1):
        query = urlencode({**filters, "page": page})
        payload = json.loads(fetch(f"https://api.inaturalist.org/v1/observations?{query}"))
        records.extend(payload["results"])
        if len(records) >= payload["total_results"] or (
            goal and len({r["user"]["login"] for r in records} - set(excluded_observers)) >= goal
        ):
            break
        time.sleep(.8)

    random.Random(SEED).shuffle(records)
    users, samples = set(excluded_observers), []
    for observation in records:
        observer = observation["user"]["login"]
        taxon = observation["taxon"]
        ancestors = [int(item) for item in (taxon.get("ancestry") or "").split("/") if item]
        if taxon_id not in [taxon["id"], *ancestors]:
            continue
        observed = observation.get("observed_on") or ""
        if observer in users or not start <= observed <= end:
            continue
        for photo in observation["photos"]:
            license_name = photo["license_code"]
            url = photo.get("url") or ""
            if license_name in LICENSES and url.startswith(
                "https://inaturalist-open-data.s3.amazonaws.com/photos/"
            ):
                url = url.replace("/square.", "/medium.")
                path = folder / class_name / f"{photo['id']}.jpg"
                samples.append({
                    "path": path.relative_to(ROOT).as_posix(), "label": class_name,
                    "observation": f"https://www.inaturalist.org/observations/{observation['id']}",
                    "date_observed": observation["observed_on"], "taxon": taxon["name"],
                    "photo": url, "license": license_name, "attribution": photo["attribution"],
                    "observer": observer,
                })
                users.add(observer)
                break
    return samples


def download(entry):
    path = ROOT / entry["path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".download")
        temp.write_bytes(fetch(entry["photo"]))
        with Image.open(temp) as image:
            image.load()
        temp.replace(path)
    with Image.open(path) as image:
        image.load()
    return {**entry, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_external():
    locked_manifest = RESULTS / "external_manifest.csv"
    if locked_manifest.exists():
        records = pd.read_csv(locked_manifest).to_dict(orient="records")
        with ThreadPoolExecutor(max_workers=6) as pool:
            for expected, downloaded in zip(records, pool.map(download, records), strict=True):
                if expected["sha256"] != downloaded["sha256"]:
                    raise ValueError(f"External image changed since the holdout was fixed: {expected['path']}")
        print(f"Verified all {len(records)} images in the fixed external holdout.", flush=True)
        return
    samples = []
    for label, taxon in TAXA.items():
        candidates = sample_observations(label, taxon)
        if not candidates:
            raise ValueError(f"No qualifying iNaturalist observations found for {label}.")
        with ThreadPoolExecutor(max_workers=6) as pool:
            batch = list(pool.map(download, candidates[:MAX_PER_CLASS]))
        samples.extend(batch)
        print(f"Downloaded {len(batch)} dated {label} photos.", flush=True)

    metadata = pd.DataFrame(samples)
    raw_paths = sorted(path for path in RAW.rglob("*") if path.suffix.lower() in EXTENSIONS)
    external_paths = [ROOT / path for path in metadata.path]
    inventory, fingerprints, unreadable = scan_images(
        raw_paths + external_paths, lambda p: p.parent.name, lambda p: p.relative_to(ROOT).as_posix()
    )
    raw_count = len(inventory) - len(samples)
    if any(entry["path"].startswith(EXTERNAL.relative_to(ROOT).as_posix()) for entry in unreadable):
        raise ValueError("An external photo did not decode.")
    pairs = list(related_pairs(inventory, fingerprints))
    groups = create_groups(len(inventory), pairs)
    raw_groups = set(groups[:raw_count])
    seen = set(raw_groups)
    excluded, keep = [], []
    for i, group in enumerate(groups[raw_count:]):
        if group in seen:
            excluded.append(i)
        else:
            keep.append(i)
            seen.add(group)
    metadata.iloc[excluded].to_csv(RESULTS / "external_duplicate_exclusions.csv", index=False)
    retained = metadata.iloc[keep]
    if set(retained.label) != set(TAXA):
        raise ValueError("All five labels must be represented in the external holdout.")
    retained.to_csv(locked_manifest, index=False)
    print(f"Locked holdout: {len(retained)} photos, {len(excluded)} overlap exclusions.", flush=True)


if __name__ == "__main__":
    prepare_external()
