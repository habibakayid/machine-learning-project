"""Flower image prediction using the three trained coursework models.

The feature extraction below matches flowers_recognition.ipynb.
"""

from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog


IMAGE_SIZE = (128, 128)
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "knn": "knn.joblib",
    "kmeans": "kmeans.joblib",
}


def load_models(model_dir=MODEL_DIR):
    """Load the saved models and their preprocessing information once."""
    model_dir = Path(model_dir)
    models = {}
    for name, filename in MODEL_FILES.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        models[name] = joblib.load(path)
    return models


def open_image(image_input):
    """Read an image path, uploaded file, or PIL Image as RGB."""
    if isinstance(image_input, Image.Image):
        return ImageOps.exif_transpose(image_input).convert("RGB")
    with Image.open(image_input) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def extract_features(image_input):
    """Same image features and order used in the training notebook."""
    image = open_image(image_input)
    rgb = np.asarray(
        image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    raw = np.asarray(
        image.resize((24, 24), Image.Resampling.BILINEAR), dtype=np.float32
    ).ravel() / 255.0
    shape = hog(
        rgb2gray(rgb), orientations=9,
        pixels_per_cell=(16, 16), cells_per_block=(2, 2)
    )

    hsv = rgb2hsv(rgb)
    height, width = hsv.shape[:2]
    row_mid, col_mid = height // 2, width // 2
    parts = [
        hsv,
        hsv[height // 4:3 * height // 4, width // 4:3 * width // 4],
        hsv[:row_mid, :col_mid],
        hsv[:row_mid, col_mid:],
        hsv[row_mid:, :col_mid],
        hsv[row_mid:, col_mid:],
    ]
    color = []
    for part in parts:
        hue, sat, value = part.reshape(-1, 3).T
        hs_hist, _, _ = np.histogram2d(
            hue, sat, bins=(18, 8), range=((0, 1), (0, 1))
        )
        value_hist, _ = np.histogram(value, bins=12, range=(0, 1))
        color.extend([hs_hist.ravel() / len(hue), value_hist / len(hue)])
    return np.concatenate([raw, shape, *color]).astype(np.float32)


def _predict_from_features(features, name, saved):
    """Apply the correct saved preprocessing and return the flower name."""
    if name == "logistic_regression":
        hog_features = saved["hog_pca"].transform(
            saved["hog_scaler"].transform(
                features[:, saved["raw_end"]:saved["hog_end"]]
            )
        )
        color_features = saved["color_pca"].transform(
            saved["color_scaler"].transform(features[:, saved["hog_end"]:])
        )
        prepared = np.hstack([hog_features, color_features])
        prediction = saved["model"].predict(prepared)

    elif name == "knn":
        prepared = features[:, saved["hog_end"]:]
        prediction = saved["model"].predict(prepared)

    elif name == "kmeans":
        prepared = saved["raw_pca"].transform(
            saved["raw_scaler"].transform(features[:, :saved["raw_end"]])
        )
        cluster_id = saved["model"].predict(prepared)[0]
        prediction = np.array([saved["cluster_mapping"][cluster_id]])

    else:
        raise ValueError(f"Unknown model: {name}. Choose one of {list(MODEL_FILES)}")

    return str(saved["encoder"].inverse_transform(prediction)[0])


def predict_flower(image_input, model_name="logistic_regression", models=None):
    """Predict one flower class using the selected trained model."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Choose one of: {', '.join(MODEL_FILES)}")
    if models is None:
        models = load_models()
    features = extract_features(image_input).reshape(1, -1)
    return _predict_from_features(features, model_name, models[model_name])


def predict_all(image_input, models=None):
    """Run all three models on the same new flower image."""
    if models is None:
        models = load_models()
    features = extract_features(image_input).reshape(1, -1)
    results = {}
    for name in MODEL_FILES:
        results[name] = _predict_from_features(features, name, models[name])
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python predict_flower.py path/to/flower.jpg")
    else:
        saved_models = load_models()
        for name, flower in predict_all(sys.argv[1], saved_models).items():
            print(f"{name}: {flower}")
