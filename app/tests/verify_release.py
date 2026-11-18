"""Verify frozen release hashes and original-project inference expectations."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from inference import FILES, predict_image


def main():
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "MODEL_MANIFEST.json").read_text())
    for model in manifest:
        assert model["fit_counts"] == {"kaggle": 3017}
        assert sha(ROOT / model["path"]) == model["sha256"]
    for sample in json.loads((ROOT / "samples/MANIFEST.json").read_text()):
        assert sha(ROOT / sample["path"]) == sample["sha256"]
    payload = ROOT / "SHA256SUMS.json"
    if payload.exists():
        for path, expected in json.loads(payload.read_text()).items():
            assert sha(ROOT / path) == expected, path
    bundles = {name: joblib.load(ROOT / "models" / file) for name, file in FILES.items()}
    cases = []
    for fixture in json.loads((ROOT / "tests/fixtures.json").read_text()):
        with Image.open(ROOT / fixture["path"]) as image:
            label, probabilities, cluster = predict_image(bundles[fixture["family"]], image)
        assert label == fixture["predicted"]
        assert (None if cluster is None else cluster + 1) == fixture["cluster_display"]
        if probabilities is not None:
            assert set(probabilities) == set(fixture["probabilities"])
            assert all(np.isclose(value, fixture["probabilities"][name], rtol=1e-7, atol=1e-9)
                       for name, value in probabilities.items())
        cases.append({"model": fixture["family"], "actual": fixture["actual"],
                      "prediction": label, "status": "passed"})
    assert len(cases) == 15
    result = {"result": "passed", "model_hashes_verified": 3,
              "original_inference_expectations_matched": 15, "cases": cases,
              "python": sys.version.split()[0], "model_training_performed": False}
    (ROOT / "evidence/RELEASE_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: 3 unchanged bundles; 15 predictions, probabilities / clusters match original inference.")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
