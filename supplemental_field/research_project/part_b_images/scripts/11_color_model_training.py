
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
    "part_b_images/data/processed/features"
)

RESULTS_PATH = Path("part_b_images/results")
MODELS_PATH = Path("models")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. Load Color Features
# =====================================

with np.load(FEATURES_PATH / "train_color.npz") as data:
    X_train = data["X"]
    y_train = data["y"]

with np.load(FEATURES_PATH / "validation_color.npz") as data:
    X_val = data["X"]
    y_val = data["y"]

print("\n===== COLOR FEATURES DATA =====")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# =====================================
# 3. Training Function
# =====================================

def train_and_tune(model_name, param_values):

    best_model = None
    best_param = None
    best_f1 = -1
    results = []

    for param in param_values:

        print(f"\nTraining {model_name}: Parameter={param}")

        if model_name == "Logistic Regression":

            classifier = LogisticRegression(
                C=param,
                max_iter=1000,
                solver="lbfgs",
                random_state=42
            )

        else:

            classifier = KNeighborsClassifier(
                n_neighbors=param,
                weights="distance",
                metric="euclidean",
                n_jobs=1
            )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier)
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)

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
            "parameter": param,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_param = param
            best_model = model

    return best_model, best_param, results


# =====================================
# 4. Logistic Regression
# =====================================

print("\n===== LOGISTIC REGRESSION =====")

lr_model, best_C, lr_results = train_and_tune(
    "Logistic Regression",
    [0.01, 0.1, 1.0, 10.0]
)

print(f"\nBest C: {best_C}")

print(classification_report(
    y_val,
    lr_model.predict(X_val),
    zero_division=0
))

joblib.dump(
    lr_model,
    MODELS_PATH / "part_b_logistic_color.joblib"
)

# =====================================
# 5. KNN Classifier
# =====================================

print("\n===== KNN CLASSIFIER =====")

knn_model, best_k, knn_results = train_and_tune(
    "KNN",
    [3, 5, 9, 15]
)

print(f"\nBest k: {best_k}")

print(classification_report(
    y_val,
    knn_model.predict(X_val),
    zero_division=0
))

joblib.dump(
    knn_model,
    MODELS_PATH / "part_b_knn_color.joblib"
)

# =====================================
# 6. Save Results
# =====================================

all_results = pd.DataFrame(
    lr_results + knn_results
)

all_results.to_csv(
    RESULTS_PATH / "color_validation_results.csv",
    index=False
)

# =====================================
# 7. Compare With HOG Models
# =====================================

comparison = []

baseline_files = [
    ("Logistic Regression", "logistic_validation_results.csv"),
    ("KNN", "knn_validation_results.csv")
]

for model_name, filename in baseline_files:

    baseline = pd.read_csv(RESULTS_PATH / filename)

    best_row = baseline.loc[
        baseline["validation_macro_f1"].idxmax()
    ]

    comparison.append({
        "model": model_name,
        "features": "HOG",
        "accuracy": best_row["validation_accuracy"],
        "macro_f1": best_row["validation_macro_f1"]
    })

for model_name in ["Logistic Regression", "KNN"]:

    model_results = all_results[
        all_results["model"] == model_name
    ]

    best_row = model_results.loc[
        model_results["macro_f1"].idxmax()
    ]

    comparison.append({
        "model": model_name,
        "features": "Color Histogram",
        "accuracy": best_row["accuracy"],
        "macro_f1": best_row["macro_f1"]
    })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    RESULTS_PATH / "hog_vs_color_comparison.csv",
    index=False
)

print("\n===== HOG VS COLOR COMPARISON =====")
print(comparison_df.to_string(index=False))

print("\nColor Model Training Completed!")
