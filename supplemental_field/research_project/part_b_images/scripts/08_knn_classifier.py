from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
# 2. Load Training and Validation Data
# =====================================

with np.load(FEATURES_PATH / "train_hog.npz") as data:
    X_train = data["X"]
    y_train = data["y"]

with np.load(FEATURES_PATH / "validation_hog.npz") as data:
    X_val = data["X"]
    y_val = data["y"]

print("\n===== DATA INFORMATION =====")
print("Training Features:", X_train.shape)
print("Validation Features:", X_val.shape)

# =====================================
# 3. Hyperparameter Tuning
# =====================================

k_values = [3, 5, 9]

results = []

best_model = None
best_k = None
best_f1 = -1

for k in k_values:

    print(f"\nTraining KNN with k={k}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
            n_jobs=1
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro"
    )

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")

    results.append({
        "k": k,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1
    })

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_k = k
        best_model = model

# =====================================
# 4. Save Tuning Results
# =====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_PATH / "knn_validation_results.csv",
    index=False
)

# =====================================
# 5. Best Model Validation Report
# =====================================

best_predictions = best_model.predict(X_val)

print("\n===== BEST KNN CLASSIFIER =====")
print(f"Best k: {best_k}")
print(f"Best Validation Macro F1: {best_f1:.4f}")

print("\n===== VALIDATION CLASSIFICATION REPORT =====")

print(classification_report(
    y_val,
    best_predictions,
    zero_division=0
))

# =====================================
# 6. Save Best Model
# =====================================

model_file = MODELS_PATH / "part_b_knn_classifier.joblib"

joblib.dump(best_model, model_file)

print(f"\nModel saved to: {model_file}")
print("\nKNN Training Completed!")
