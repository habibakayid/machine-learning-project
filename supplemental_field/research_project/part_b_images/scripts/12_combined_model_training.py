
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

FEATURES_PATH = Path(
    "part_b_images/data/processed/features"
)

RESULTS_PATH = Path("part_b_images/results")
MODELS_PATH = Path("models")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. Load Features
# =====================================

def load_features(split_name, feature_type):

    file_path = (
        FEATURES_PATH / f"{split_name}_{feature_type}.npz"
    )

    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]

    return X, y


# Training
X_train_hog, y_train = load_features("train", "hog")
X_train_color, y_train_color = load_features(
    "train", "color"
)

# Validation
X_val_hog, y_val = load_features("validation", "hog")
X_val_color, y_val_color = load_features(
    "validation", "color"
)

# Confirm that labels have the same order
assert np.array_equal(y_train, y_train_color)
assert np.array_equal(y_val, y_val_color)

# =====================================
# 3. Combine Features
# =====================================

X_train = np.hstack(
    (X_train_hog, X_train_color)
)

X_val = np.hstack(
    (X_val_hog, X_val_color)
)

hog_size = X_train_hog.shape[1]
color_size = X_train_color.shape[1]

print("\n===== COMBINED FEATURES =====")
print("HOG Features:", hog_size)
print("Color Features:", color_size)
print("Total Features:", X_train.shape[1])
print("Training Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)

# =====================================
# 4. Create Model Pipeline
# =====================================

def create_pipeline(model_name, parameter):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "hog",
                StandardScaler(),
                slice(0, hog_size)
            ),
            (
                "color",
                StandardScaler(),
                slice(hog_size, hog_size + color_size)
            )
        ],
        transformer_weights={
            "hog": 1 / np.sqrt(hog_size),
            "color": 1 / np.sqrt(color_size)
        }
    )

    if model_name == "Logistic Regression":

        classifier = LogisticRegression(
            C=parameter,
            max_iter=1500,
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
# 5. Train and Tune
# =====================================

experiments = {
    "Logistic Regression": [0.01, 0.1, 1.0, 10.0],
    "KNN": [5, 9, 15, 21, 31]
}

all_results = []
best_results = []

for model_name, parameters in experiments.items():

    print(f"\n===== {model_name.upper()} =====")

    best_model = None
    best_param = None
    best_f1 = -1

    for parameter in parameters:

        print(f"\nTrying parameter = {parameter}")

        model = create_pipeline(
            model_name, parameter
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
            "parameter": parameter,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_param = parameter
            best_model = model

    # Best model for this algorithm
    best_predictions = best_model.predict(X_val)

    print(f"\n===== BEST {model_name.upper()} =====")
    print("Best Parameter:", best_param)
    print(f"Best Macro F1: {best_f1:.4f}")

    print(classification_report(
        y_val,
        best_predictions,
        zero_division=0
    ))

    model_filename = (
        "part_b_logistic_combined.joblib"
        if model_name == "Logistic Regression"
        else "part_b_knn_combined.joblib"
    )

    joblib.dump(
        best_model,
        MODELS_PATH / model_filename
    )

    best_results.append({
        "model": model_name,
        "features": "HOG + Color",
        "accuracy": accuracy_score(
            y_val, best_predictions
        ),
        "macro_f1": best_f1
    })

# =====================================
# 6. Save Results
# =====================================

pd.DataFrame(all_results).to_csv(
    RESULTS_PATH / "combined_validation_results.csv",
    index=False
)

# =====================================
# 7. Compare All Feature Methods
# =====================================

previous = pd.read_csv(
    RESULTS_PATH / "hog_vs_color_comparison.csv"
)

combined = pd.DataFrame(best_results)

comparison = pd.concat(
    [previous, combined],
    ignore_index=True
)

comparison.to_csv(
    RESULTS_PATH / "all_feature_comparison.csv",
    index=False
)

print("\n===== FINAL FEATURE COMPARISON =====")
print(comparison.to_string(index=False))

print("\nCombined Feature Training Completed!")
