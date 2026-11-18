
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

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
    "part_b_images/data/processed/features_spatial"
)

RESULTS_PATH = Path("part_b_images/results")
MODELS_PATH = Path("models")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. Load Training and Validation Data
# =====================================

with np.load(
    FEATURES_PATH / "train_spatial_hsv.npz"
) as data:
    X_train = data["X"]
    y_train = data["y"]

with np.load(
    FEATURES_PATH / "validation_spatial_hsv.npz"
) as data:
    X_val = data["X"]
    y_val = data["y"]

print("\n===== DATA INFORMATION =====")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)

assert X_train.shape[1] == 240
assert X_val.shape[1] == 240

# =====================================
# 3. Model Training Function
# =====================================

def train_and_tune(model_name, parameters):

    best_model = None
    best_parameter = None
    best_f1 = -1

    results = []

    for parameter in parameters:

        print(f"\nTrying {model_name}: {parameter}")

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
                metric="euclidean"
            )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier)
        ])

        # Training
        model.fit(X_train, y_train)

        # Validation predictions
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

        results.append({
            "model": model_name,
            "parameter": parameter,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        # Select best validation model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_parameter = parameter
            best_model = model

    # Best model report
    best_predictions = best_model.predict(X_val)

    best_accuracy = accuracy_score(
        y_val, best_predictions
    )

    print(f"\n===== BEST {model_name.upper()} =====")
    print("Best Parameter:", best_parameter)
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Macro F1: {best_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        best_predictions,
        zero_division=0
    ))

    return best_model, results


# =====================================
# 4. Logistic Regression Training
# =====================================

print("\n===== LOGISTIC REGRESSION =====")

lr_model, lr_results = train_and_tune(
    "Logistic Regression",
    [0.01, 0.1, 1.0, 10.0, 100.0]
)

joblib.dump(
    lr_model,
    MODELS_PATH / "part_b_logistic_spatial_hsv.joblib"
)

# =====================================
# 5. KNN Training
# =====================================

print("\n===== KNN CLASSIFIER =====")

knn_model, knn_results = train_and_tune(
    "KNN",
    [3, 5, 9, 15, 21, 31]
)

joblib.dump(
    knn_model,
    MODELS_PATH / "part_b_knn_spatial_hsv.joblib"
)

# =====================================
# 6. Save All Results
# =====================================

results_df = pd.DataFrame(
    lr_results + knn_results
)

results_df.to_csv(
    RESULTS_PATH / "spatial_hsv_validation_results.csv",
    index=False
)

# =====================================
# 7. Compare Best Models
# =====================================

best_results = []

for model_name in ["Logistic Regression", "KNN"]:

    model_results = results_df[
        results_df["model"] == model_name
    ]

    best_row = model_results.loc[
        model_results["macro_f1"].idxmax()
    ]

    best_results.append({
        "model": model_name,
        "features": "Spatial HSV",
        "accuracy": best_row["accuracy"],
        "macro_f1": best_row["macro_f1"]
    })

comparison = pd.DataFrame(best_results)

print("\n===== SPATIAL HSV COMPARISON =====")
print(comparison.to_string(index=False))

print("\nPrevious Best Accuracy: 0.5650")
print("Previous Best Model: Logistic Regression")
print("Previous Features: HOG + RGB Color")

print("\nSpatial HSV Training Completed!")
