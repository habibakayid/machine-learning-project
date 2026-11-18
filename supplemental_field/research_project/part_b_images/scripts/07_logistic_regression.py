
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
print("Training Labels:", y_train.shape)
print("Validation Labels:", y_val.shape)

# =====================================
# 3. Hyperparameter Tuning
# =====================================

C_values = [0.1, 1.0, 10.0]

results = []

best_model = None
best_C = None
best_f1 = -1

for C in C_values:

    print(f"\nTraining Logistic Regression with C={C}")

    # Pipeline prevents data leakage:
    # StandardScaler is fitted on training data only.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predictions on validation data
    y_pred = model.predict(X_val)

    # Validation metrics
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro"
    )

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")

    results.append({
        "C": C,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1
    })

    # Select best model using Macro F1
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_C = C
        best_model = model

# =====================================
# 4. Save Tuning Results
# =====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_PATH / "logistic_validation_results.csv",
    index=False
)

# =====================================
# 5. Evaluate Best Model on Validation
# =====================================

best_predictions = best_model.predict(X_val)

print("\n===== BEST LOGISTIC REGRESSION =====")
print(f"Best C: {best_C}")
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

model_file = MODELS_PATH / "part_b_logistic_regression.joblib"

joblib.dump(best_model, model_file)

print(f"\nModel saved to: {model_file}")
print("\nLogistic Regression Training Completed!")
