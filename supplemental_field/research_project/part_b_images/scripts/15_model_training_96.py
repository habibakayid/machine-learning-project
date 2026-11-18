
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
    "part_b_images/data/processed/features_96"
)

RESULTS_PATH = Path("part_b_images/results")
MODELS_PATH = Path("models")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. Load Features
# =====================================

def load_features(split, feature_type):

    file_path = (
        FEATURES_PATH / f"{split}_{feature_type}.npz"
    )

    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]

    return X, y


X_train_hog, y_train = load_features("train", "hog")
X_train_color, y_train_color = load_features(
    "train", "color"
)

X_val_hog, y_val = load_features("validation", "hog")
X_val_color, y_val_color = load_features(
    "validation", "color"
)

# Check label alignment
assert np.array_equal(y_train, y_train_color)
assert np.array_equal(y_val, y_val_color)

# Combine HOG and Color Features
X_train = np.hstack([X_train_hog, X_train_color])
X_val = np.hstack([X_val_hog, X_val_color])

hog_size = X_train_hog.shape[1]
color_size = X_train_color.shape[1]

print("\n===== DATA INFORMATION =====")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("HOG Features:", hog_size)
print("Color Features:", color_size)


# =====================================
# 3. Create Pipeline
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
# 4. Hyperparameter Tuning
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
    best_parameter = None
    best_f1 = -1

    for parameter in parameters:

        print(f"\nTrying parameter = {parameter}")

        model = create_pipeline(
            model_name,
            parameter
        )

        # Train on training data only
        model.fit(X_train, y_train)

        # Predict validation data
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

        all_results.append({
            "model": model_name,
            "image_size": 96,
            "parameter": parameter,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        # Choose best model using Macro F1
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_parameter = parameter
            best_model = model

    # =====================================
    # 5. Best Model Results
    # =====================================

    best_predictions = best_model.predict(X_val)

    best_accuracy = accuracy_score(
        y_val,
        best_predictions
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

    # Save model
    if model_name == "Logistic Regression":
        filename = "part_b_logistic_96.joblib"
    else:
        filename = "part_b_knn_96.joblib"

    joblib.dump(
        best_model,
        MODELS_PATH / filename
    )

    best_results.append({
        "model": model_name,
        "image_size": "96x96",
        "accuracy": best_accuracy,
        "macro_f1": best_f1,
        "parameter": best_parameter
    })


# =====================================
# 6. Save Experiment Results
# =====================================

pd.DataFrame(all_results).to_csv(
    RESULTS_PATH / "training_96_results.csv",
    index=False
)

pd.DataFrame(best_results).to_csv(
    RESULTS_PATH / "best_models_96.csv",
    index=False
)


# =====================================
# 7. Compare with 64x64 Results
# =====================================

print("\n===== 64x64 VS 96x96 COMPARISON =====")

old_results = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "image_size": "64x64",
        "accuracy": 0.565015,
        "macro_f1": 0.559728
    },
    {
        "model": "KNN",
        "image_size": "64x64",
        "accuracy": 0.501548,
        "macro_f1": 0.491351
    }
])

new_results = pd.DataFrame(best_results)

comparison = pd.concat(
    [
        old_results,
        new_results[
            ["model", "image_size", "accuracy", "macro_f1"]
        ]
    ],
    ignore_index=True
)

print(comparison.to_string(index=False))

comparison.to_csv(
    RESULTS_PATH / "64_vs_96_comparison.csv",
    index=False
)

print("\n96x96 Model Training Completed!")
