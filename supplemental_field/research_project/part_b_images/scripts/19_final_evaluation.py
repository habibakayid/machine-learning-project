
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_score
)

# =====================================
# 1. Paths
# =====================================

BASE_PATH = Path("part_b_images/data/processed")

HOG_PATH = BASE_PATH / "features"
HSV_PATH = BASE_PATH / "features_spatial"

MODELS_PATH = Path("models")
RESULTS_PATH = Path("part_b_images/results")
FIGURES_PATH = Path("part_b_images/figures")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

# =====================================
# 2. Check Required Files
# =====================================

required_files = [
    HOG_PATH / "test_hog.npz",
    HOG_PATH / "test_color.npz",
    HSV_PATH / "test_spatial_hsv.npz",
    MODELS_PATH / "part_b_logistic_hog_hsv.joblib",
    MODELS_PATH / "part_b_knn_combined.joblib",
    MODELS_PATH / "part_b_kmeans_combined.joblib",
    MODELS_PATH / "part_b_kmeans_combined_mapping.json"
]

missing_files = [
    str(path)
    for path in required_files
    if not path.is_file()
]

if missing_files:
    raise FileNotFoundError(
        "Missing required files:\n"
        + "\n".join(missing_files)
    )

# Prevent accidental repeated test evaluation
summary_file = RESULTS_PATH / "final_test_summary.csv"

if summary_file.exists():
    raise RuntimeError(
        "Final test results already exist. "
        "Do not repeat evaluation to tune models."
    )

# =====================================
# 3. Load Test Features
# =====================================

def load_features(folder, filename):

    with np.load(
        folder / filename,
        allow_pickle=False
    ) as data:

        X = data["X"]
        y = data["y"]

    return X, y


X_hog, y_test = load_features(
    HOG_PATH, "test_hog.npz"
)

X_rgb, y_rgb = load_features(
    HOG_PATH, "test_color.npz"
)

X_hsv, y_hsv = load_features(
    HSV_PATH, "test_spatial_hsv.npz"
)

# Verify labels and dimensions
assert np.array_equal(y_test, y_rgb)
assert np.array_equal(y_test, y_hsv)

assert X_hog.shape[1] == 1764
assert X_rgb.shape[1] == 48
assert X_hsv.shape[1] == 240

# Combine test features
X_hog_rgb = np.hstack([X_hog, X_rgb])
X_hog_hsv = np.hstack([X_hog, X_hsv])

print("\n===== TEST DATA =====")
print("Test Images:", len(y_test))
print("HOG + RGB:", X_hog_rgb.shape)
print("HOG + HSV:", X_hog_hsv.shape)

# =====================================
# 4. Load Saved Models
# =====================================

logistic_model = joblib.load(
    MODELS_PATH / "part_b_logistic_hog_hsv.joblib"
)

knn_model = joblib.load(
    MODELS_PATH / "part_b_knn_combined.joblib"
)

kmeans_model = joblib.load(
    MODELS_PATH / "part_b_kmeans_combined.joblib"
)

with open(
    MODELS_PATH / "part_b_kmeans_combined_mapping.json",
    "r",
    encoding="utf-8"
) as file:

    cluster_mapping = json.load(file)

# =====================================
# 5. Final Predictions
# =====================================

logistic_predictions = logistic_model.predict(
    X_hog_hsv
)

knn_predictions = knn_model.predict(
    X_hog_rgb
)

# K-Means predicts cluster IDs
kmeans_clusters = kmeans_model.predict(
    X_hog_rgb
)

# Map cluster IDs to flower labels
kmeans_predictions = np.array([
    cluster_mapping[str(int(cluster))]
    for cluster in kmeans_clusters
])

models = {
    "Logistic Regression": logistic_predictions,
    "KNN Classifier": knn_predictions,
    "K-Means": kmeans_predictions
}

# =====================================
# 6. Evaluation Metrics
# =====================================

summary_results = []
class_results = []

for model_name, predictions in models.items():

    accuracy = accuracy_score(
        y_test, predictions
    )

    precision = precision_score(
        y_test, predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test, predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    print(f"\n===== {model_name.upper()} =====")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        zero_division=0,
        output_dict=True
    )

    print(classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        zero_division=0
    ))

    summary_results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1
    })

    for class_name in CLASSES:

        metrics = report[class_name]

        class_results.append({
            "model": model_name,
            "class": class_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": metrics["support"]
        })

    # =====================================
    # 7. Confusion Matrix
    # =====================================

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASSES
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        xticks_rotation=45
    )

    ax.set_title(
        f"{model_name} - Test Confusion Matrix"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    fig.savefig(
        FIGURES_PATH / f"{filename}_test_confusion.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# =====================================
# 8. K-Means Silhouette Score
# =====================================

# Transform test features using the saved
# training-fitted preprocessing and PCA
X_kmeans_pca = kmeans_model[:-1].transform(
    X_hog_rgb
)

if len(np.unique(kmeans_clusters)) > 1:

    silhouette = silhouette_score(
        X_kmeans_pca,
        kmeans_clusters
    )

else:
    silhouette = float("nan")

print("\n===== K-MEANS DETAILS =====")

print("Cluster Mapping:")
for cluster_id, label in sorted(cluster_mapping.items()):
    print(f"Cluster {cluster_id} -> {label}")

print(f"Test Silhouette Score: {silhouette:.4f}")

# Save cluster analysis
cluster_df = pd.DataFrame({
    "cluster": kmeans_clusters,
    "true_label": y_test,
    "predicted_label": kmeans_predictions
})

cluster_table = pd.crosstab(
    cluster_df["cluster"],
    cluster_df["true_label"]
)

cluster_table.to_csv(
    RESULTS_PATH / "kmeans_test_clusters.csv"
)

# =====================================
# 9. Save Final Results
# =====================================

summary_df = pd.DataFrame(summary_results)

summary_df.to_csv(
    summary_file,
    index=False
)

pd.DataFrame(class_results).to_csv(
    RESULTS_PATH / "final_test_per_class.csv",
    index=False
)

print("\n===== FINAL TEST COMPARISON =====")

print(summary_df.to_string(index=False))

# =====================================
# 10. Final Comparison Chart
# =====================================

x = np.arange(len(summary_df))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width / 2,
    summary_df["accuracy"] * 100,
    width,
    label="Accuracy"
)

ax.bar(
    x + width / 2,
    summary_df["macro_f1"] * 100,
    width,
    label="Macro F1"
)

ax.set_xticks(x)
ax.set_xticklabels(
    summary_df["model"]
)

ax.set_ylabel("Score (%)")
ax.set_title("Final Test Model Comparison")
ax.set_ylim(0, 100)
ax.legend()

plt.tight_layout()

fig.savefig(
    FIGURES_PATH / "final_test_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nFinal Evaluation Completed Successfully!")
