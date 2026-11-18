
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# =====================================
# 1. Paths
# =====================================

BASE_PATH = Path("part_b_images/data/processed")

FEATURES_PATH = BASE_PATH / "features"
HSV_PATH = BASE_PATH / "features_spatial"

RESULTS_PATH = Path("part_b_images/results")
MODELS_PATH = Path("models")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. Load Features
# =====================================

def load_features(folder, filename):

    with np.load(
        folder / filename,
        allow_pickle=False
    ) as data:

        X = data["X"]
        y = data["y"]

    return X, y


# Training features
X_train_hog, y_train = load_features(
    FEATURES_PATH, "train_hog.npz"
)

X_train_rgb, y_train_rgb = load_features(
    FEATURES_PATH, "train_color.npz"
)

X_train_hsv, y_train_hsv = load_features(
    HSV_PATH, "train_spatial_hsv.npz"
)


# Validation features
X_val_hog, y_val = load_features(
    FEATURES_PATH, "validation_hog.npz"
)

X_val_rgb, y_val_rgb = load_features(
    FEATURES_PATH, "validation_color.npz"
)

X_val_hsv, y_val_hsv = load_features(
    HSV_PATH, "validation_spatial_hsv.npz"
)


# =====================================
# 3. Verify Data
# =====================================

assert np.array_equal(y_train, y_train_rgb)
assert np.array_equal(y_train, y_train_hsv)

assert np.array_equal(y_val, y_val_rgb)
assert np.array_equal(y_val, y_val_hsv)

assert X_train_hog.shape[1] == 1764
assert X_train_rgb.shape[1] == 48
assert X_train_hsv.shape[1] == 240

# Features must also have the same image order
# in all three files.


# =====================================
# 4. Combine Features
# =====================================

X_train = np.hstack([
    X_train_hog,
    X_train_rgb,
    X_train_hsv
])

X_val = np.hstack([
    X_val_hog,
    X_val_rgb,
    X_val_hsv
])

hog_size = X_train_hog.shape[1]
rgb_size = X_train_rgb.shape[1]
hsv_size = X_train_hsv.shape[1]

print("\n===== COMBINED FEATURES =====")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("HOG:", hog_size)
print("RGB:", rgb_size)
print("Spatial HSV:", hsv_size)
print("Total:", X_train.shape[1])


# =====================================
# 5. Create Model Pipeline
# =====================================

def create_model(c_value, rgb_weight, hsv_weight):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "hog",
                StandardScaler(),
                slice(0, hog_size)
            ),
            (
                "rgb",
                StandardScaler(),
                slice(
                    hog_size,
                    hog_size + rgb_size
                )
            ),
            (
                "hsv",
                StandardScaler(),
                slice(
                    hog_size + rgb_size,
                    hog_size + rgb_size + hsv_size
                )
            )
        ],
        transformer_weights={
            "hog": 1 / np.sqrt(hog_size),
            "rgb": rgb_weight / np.sqrt(rgb_size),
            "hsv": hsv_weight / np.sqrt(hsv_size)
        }
    )

    classifier = LogisticRegression(
        C=c_value,
        max_iter=2000,
        solver="lbfgs",
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


# =====================================
# 6. Hyperparameter Experiments
# =====================================

c_values = [0.1, 1.0]
rgb_weights = [0.5, 1.0]
hsv_weights = [1.0, 2.0]

all_results = []

best_model = None
best_f1 = -1
best_accuracy = -1
best_config = None

print("\n===== TRAINING EXPERIMENTS =====")

for c_value in c_values:

    for rgb_weight in rgb_weights:

        for hsv_weight in hsv_weights:

            print(
                f"\nC={c_value}, "
                f"RGB Weight={rgb_weight}, "
                f"HSV Weight={hsv_weight}"
            )

            model = create_model(
                c_value,
                rgb_weight,
                hsv_weight
            )

            # Fit on training data only
            model.fit(X_train, y_train)

            # Validation prediction
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
                "C": c_value,
                "rgb_weight": rgb_weight,
                "hsv_weight": hsv_weight,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

            # Select best model using Macro F1
            if (
                macro_f1 > best_f1
                or (
                    np.isclose(macro_f1, best_f1)
                    and accuracy > best_accuracy
                )
            ):

                best_f1 = macro_f1
                best_accuracy = accuracy
                best_model = model

                best_config = {
                    "C": c_value,
                    "rgb_weight": rgb_weight,
                    "hsv_weight": hsv_weight
                }


# =====================================
# 7. Best Model Report
# =====================================

best_predictions = best_model.predict(X_val)

print("\n===== BEST MODEL =====")

print("Best C:", best_config["C"])
print("Best RGB Weight:", best_config["rgb_weight"])
print("Best HSV Weight:", best_config["hsv_weight"])

print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best Macro F1: {best_f1:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_val,
    best_predictions,
    zero_division=0
))


# =====================================
# 8. Save Model
# =====================================

model_file = (
    MODELS_PATH /
    "part_b_logistic_hog_rgb_hsv.joblib"
)

joblib.dump(best_model, model_file)

print("\nModel saved:", model_file)


# =====================================
# 9. Save Experiment Results
# =====================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["macro_f1", "accuracy"],
    ascending=False
)

results_df.to_csv(
    RESULTS_PATH / "hog_rgb_hsv_results.csv",
    index=False
)


# =====================================
# 10. Compare Previous Best
# =====================================

previous_accuracy = 0.5805
previous_f1 = 0.5759

print("\n===== MODEL COMPARISON =====")

print(f"Previous Accuracy: {previous_accuracy:.4f}")
print(f"New Accuracy:      {best_accuracy:.4f}")

print(f"Previous Macro F1: {previous_f1:.4f}")
print(f"New Macro F1:      {best_f1:.4f}")

accuracy_difference = (
    best_accuracy - previous_accuracy
) * 100

print(
    f"Accuracy Difference: "
    f"{accuracy_difference:+.2f} percentage points"
)

if best_f1 > previous_f1:

    print("\nNew model improved Validation Macro F1!")

else:

    print("\nPrevious model remains better by Macro F1.")


print("\nTraining Completed!")
