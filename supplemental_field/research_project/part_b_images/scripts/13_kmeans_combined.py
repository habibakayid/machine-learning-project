
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
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
# 2. Load HOG and Color Features
# =====================================

def load_features(split, feature_type):
    with np.load(
        FEATURES_PATH / f"{split}_{feature_type}.npz"
    ) as data:
        return data["X"], data["y"]

X_train_hog, y_train = load_features("train", "hog")
X_train_color, y_train_color = load_features("train", "color")

X_val_hog, y_val = load_features("validation", "hog")
X_val_color, y_val_color = load_features("validation", "color")

assert np.array_equal(y_train, y_train_color)
assert np.array_equal(y_val, y_val_color)

X_train = np.hstack([X_train_hog, X_train_color])
X_val = np.hstack([X_val_hog, X_val_color])

hog_size = X_train_hog.shape[1]
color_size = X_train_color.shape[1]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# =====================================
# 3. Feature Scaling
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        ("hog", StandardScaler(), slice(0, hog_size)),
        ("color", StandardScaler(),
         slice(hog_size, hog_size + color_size))
    ],
    transformer_weights={
        "hog": 1 / np.sqrt(hog_size),
        "color": 1 / np.sqrt(color_size)
    }
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)

# =====================================
# 4. PCA and K-Means Tuning
# =====================================

pca_values = [30, 60, 100]
n_init_values = [10, 20]

results = []
best_f1 = -1

best_pca = None
best_kmeans = None
best_mapping = None
best_config = None

for components in pca_values:

    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=42
    )

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    explained = pca.explained_variance_ratio_.sum()

    for n_init in n_init_values:

        print(
            f"\nPCA={components}, n_init={n_init}"
        )

        kmeans = KMeans(
            n_clusters=5,
            n_init=n_init,
            max_iter=300,
            random_state=42
        )

        train_clusters = kmeans.fit_predict(X_train_pca)

        # Map clusters using TRAIN labels only
        mapping = {}

        for cluster_id in range(5):
            labels = y_train[train_clusters == cluster_id]

            mapping[str(cluster_id)] = str(
                Counter(labels).most_common(1)[0][0]
            )

        val_clusters = kmeans.predict(X_val_pca)

        predictions = np.array([
            mapping[str(cluster)]
            for cluster in val_clusters
        ])

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        )

        silhouette = silhouette_score(
            X_train_pca,
            train_clusters,
            sample_size=500,
            random_state=42
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Silhouette: {silhouette:.4f}")
        print(f"Explained Variance: {explained:.4f}")

        results.append({
            "pca_components": components,
            "n_init": n_init,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "silhouette": silhouette,
            "explained_variance": explained
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_pca = pca
            best_kmeans = kmeans
            best_mapping = mapping.copy()
            best_config = (components, n_init)

# =====================================
# 5. Save Results
# =====================================

pd.DataFrame(results).to_csv(
    RESULTS_PATH / "kmeans_combined_validation.csv",
    index=False
)

# =====================================
# 6. Best Model Validation Report
# =====================================

X_val_best = best_pca.transform(X_val_scaled)
val_clusters = best_kmeans.predict(X_val_best)

predictions = np.array([
    best_mapping[str(cluster)]
    for cluster in val_clusters
])

print("\n===== BEST COMBINED K-MEANS =====")
print("Best PCA Components:", best_config[0])
print("Best n_init:", best_config[1])
print(f"Best Macro F1: {best_f1:.4f}")

print("\n===== CLUSTER MAPPING =====")
for cluster, label in sorted(best_mapping.items()):
    print(f"Cluster {cluster} -> {label}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_val,
    predictions,
    labels=["daisy", "dandelion", "rose",
            "sunflower", "tulip"],
    zero_division=0
))

# =====================================
# 7. Save Full Pipeline
# =====================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", best_pca),
    ("kmeans", best_kmeans)
])

joblib.dump(
    pipeline,
    MODELS_PATH / "part_b_kmeans_combined.joblib"
)

with open(
    MODELS_PATH / "part_b_kmeans_combined_mapping.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(best_mapping, file, indent=4)

print("\nCombined K-Means Completed!")
