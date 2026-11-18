
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# =====================================
# 1. Paths
# =====================================

BASE_PATH = Path("part_b_images/data/processed")

HOG_PATH = BASE_PATH / "features"
HSV_PATH = BASE_PATH / "features_spatial"

RESULTS_PATH = Path("part_b_images/results")
MODELS_PATH = Path("models")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. Load Features
# =====================================

def load_data(split, folder, suffix):
    with np.load(folder / f"{split}_{suffix}.npz") as data:
        return data["X"], data["y"]

X_train_hog, y_train = load_data(
    "train", HOG_PATH, "hog"
)

X_val_hog, y_val = load_data(
    "validation", HOG_PATH, "hog"
)

X_train_hsv, y_train_hsv = load_data(
    "train", HSV_PATH, "spatial_hsv"
)

X_val_hsv, y_val_hsv = load_data(
    "validation", HSV_PATH, "spatial_hsv"
)

# Ensure label order matches
assert np.array_equal(y_train, y_train_hsv)
assert np.array_equal(y_val, y_val_hsv)

assert X_train_hog.shape[1] == 1764
assert X_train_hsv.shape[1] == 240

X_train = np.hstack([X_train_hog, X_train_hsv])
X_val = np.hstack([X_val_hog, X_val_hsv])

hog_size = X_train_hog.shape[1]
hsv_size = X_train_hsv.shape[1]

print("\n===== COMBINED FEATURES =====")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# =====================================
# 3. Build Pipeline
# =====================================

def create_model(model_name, parameter, hsv_weight):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "hog",
                StandardScaler(),
                slice(0, hog_size)
            ),
            (
                "hsv",
                StandardScaler(),
                slice(hog_size, hog_size + hsv_size)
            )
        ],
        transformer_weights={
            "hog": 1 / np.sqrt(hog_size),
            "hsv": hsv_weight / np.sqrt(hsv_size)
        }
    )

    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            C=parameter,
            max_iter=2000,
            solver="lbfgs",
            random_state=42
        )
    else:
        classifier = KNeighborsClassifier(
            n_neighbors=parameter,
            weights="distance",
            metric="euclidean",
            n_jobs=1
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

# =====================================
# 4. Training Experiments
# =====================================

experiments = {
    "Logistic Regression": [0.1, 1.0, 10.0],
    "KNN": [9, 21]
}

hsv_weights = [0.5, 1.0, 2.0]

all_results = []

for model_name, parameters in experiments.items():

    best_model = None
    best_f1 = -1
    best_config = None

    print(f"\n===== {model_name.upper()} =====")

    for weight in hsv_weights:
        for parameter in parameters:

            print(
                f"\nHSV Weight={weight}, "
                f"Parameter={parameter}"
            )

            model = create_model(
                model_name,
                parameter,
                weight
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            accuracy = accuracy_score(
                y_val, predictions
            )

            macro_f1 = f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0
            )

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")

            all_results.append({
                "model": model_name,
                "hsv_weight": weight,
                "parameter": parameter,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_model = model
                best_config = (weight, parameter)

    # =====================================
    # 5. Best Model Report
    # =====================================

    best_predictions = best_model.predict(X_val)

    print(f"\n===== BEST {model_name.upper()} =====")

    print("Best HSV Weight:", best_config[0])
    print("Best Parameter:", best_config[1])

    print(
        "Best Accuracy:",
        round(accuracy_score(y_val, best_predictions), 4)
    )

    print("Best Macro F1:", round(best_f1, 4))

    print(classification_report(
        y_val,
        best_predictions,
        zero_division=0
    ))

    # Save model
    filename = (
        "part_b_logistic_hog_hsv.joblib"
        if model_name == "Logistic Regression"
        else "part_b_knn_hog_hsv.joblib"
    )

    joblib.dump(
        best_model,
        MODELS_PATH / filename
    )

# =====================================
# 6. Save Results
# =====================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    RESULTS_PATH / "hog_spatial_hsv_results.csv",
    index=False
)

print("\n===== BEST HOG + SPATIAL HSV RESULTS =====")

for model_name in experiments:

    subset = results_df[
        results_df["model"] == model_name
    ]

    best = subset.loc[subset["macro_f1"].idxmax()]

    print(
        f"{model_name}: "
        f"Accuracy={best['accuracy']:.4f}, "
        f"Macro F1={best['macro_f1']:.4f}"
    )

print("\nTraining Completed!")
