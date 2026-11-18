
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    silhouette_score
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
# 2. Load Train and Validation
# =====================================

with np.load(FEATURES_PATH / "train_hog.npz") as data:
    X_train = data["X"]
    y_train = data["y"]

with np.load(FEATURES_PATH / "validation_hog.npz") as data:
    X_val = data["X"]
    y_val = data["y"]

print("\n===== DATA INFORMATION =====")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# =====================================
# 3. Scaling and PCA
# =====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

pca = PCA(
    n_components=60,
    svd_solver="randomized",
    random_state=42
)

X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

print("\n===== PCA RESULTS =====")
print("Before PCA:", X_train.shape)
print("After PCA:", X_train_pca.shape)

print(
    "Explained Variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)

# =====================================
# 4. Hyperparameter Tuning
# =====================================

n_init_values = [5, 10, 20]

results = []

best_model = None
best_mapping = None
best_n_init = None
best_f1 = -1

for n_init in n_init_values:

    print(f"\nTraining K-Means with n_init={n_init}")

    model = KMeans(
        n_clusters=5,
        n_init=n_init,
        max_iter=250,
        random_state=42
    )

    train_clusters = model.fit_predict(X_train_pca)

    # Map each cluster to its majority training label
    cluster_mapping = {}

    for cluster_id in range(5):

        cluster_labels = y_train[
            train_clusters == cluster_id
        ]

        if len(cluster_labels) == 0:
            continue

        most_common_label = Counter(
            cluster_labels
        ).most_common(1)[0][0]

        cluster_mapping[cluster_id] = str(
            most_common_label
        )

    # Predict validation clusters
    val_clusters = model.predict(X_val_pca)

    val_predictions = np.array([
        cluster_mapping[int(cluster)]
        for cluster in val_clusters
    ])

    # Validation metrics
    accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    # Silhouette on a sample of training data
    silhouette = silhouette_score(
        X_train_pca,
        train_clusters,
        sample_size=500,
        random_state=42
    )

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")
    print(f"Train Silhouette Score: {silhouette:.4f}")

    results.append({
        "n_init": n_init,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "train_silhouette": silhouette
    })

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_n_init = n_init
        best_model = model
        best_mapping = cluster_mapping.copy()

# =====================================
# 5. Save Validation Results
# =====================================

pd.DataFrame(results).to_csv(
    RESULTS_PATH / "kmeans_validation_results.csv",
    index=False
)

# =====================================
# 6. Best Model Report
# =====================================

best_clusters = best_model.predict(X_val_pca)

best_predictions = np.array([
    best_mapping[int(cluster)]
    for cluster in best_clusters
])

print("\n===== BEST K-MEANS =====")
print(f"Best n_init: {best_n_init}")
print(f"Best Validation Macro F1: {best_f1:.4f}")

print("\n===== CLUSTER TO LABEL MAPPING =====")

for cluster_id, label in sorted(best_mapping.items()):
    print(f"Cluster {cluster_id} -> {label}")

print("\n===== VALIDATION CLASSIFICATION REPORT =====")

print(classification_report(
    y_val,
    best_predictions,
    zero_division=0
))

# =====================================
# 7. Save Model and Mapping
# =====================================

pipeline = Pipeline([
    ("scaler", scaler),
    ("pca", pca),
    ("kmeans", best_model)
])

joblib.dump(
    pipeline,
    MODELS_PATH / "part_b_kmeans.joblib"
)

with open(
    MODELS_PATH / "part_b_kmeans_mapping.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        best_mapping,
        file,
        indent=4
    )

print("\nK-Means Training Completed!")
