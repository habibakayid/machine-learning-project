
# =====================================
# Train / Validation / Test Split
# =====================================

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = Path("part_b_images/data/processed")
SPLIT_PATH = DATA_PATH / "splits"

SPLIT_PATH.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH / "clean_images.csv")

print("Total Clean Images:", len(df))

# Split features and labels
X = df["path"]
y = df["label"]

# =====================================
# Step 1: Training (70%) + Temp (30%)
# =====================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# =====================================
# Step 2: Validation (15%) + Test (15%)
# =====================================

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# =====================================
# Create DataFrames
# =====================================

train_df = pd.DataFrame({
    "path": X_train,
    "label": y_train
})

val_df = pd.DataFrame({
    "path": X_val,
    "label": y_val
})

test_df = pd.DataFrame({
    "path": X_test,
    "label": y_test
})

# =====================================
# Check for Data Leakage
# =====================================

train_paths = set(train_df["path"])
val_paths = set(val_df["path"])
test_paths = set(test_df["path"])

assert train_paths.isdisjoint(val_paths)
assert train_paths.isdisjoint(test_paths)
assert val_paths.isdisjoint(test_paths)

print("\nNo overlapping image paths detected!")

# =====================================
# Save Splits
# =====================================

train_df.to_csv(SPLIT_PATH / "train.csv", index=False)
val_df.to_csv(SPLIT_PATH / "validation.csv", index=False)
test_df.to_csv(SPLIT_PATH / "test.csv", index=False)

# =====================================
# Print Results
# =====================================

print("\n===== DATA SPLITTING REPORT =====")

print(f"Training Images: {len(train_df)}")
print(f"Validation Images: {len(val_df)}")
print(f"Testing Images: {len(test_df)}")

print("\n===== CLASS DISTRIBUTION =====")

distribution = pd.DataFrame({
    "Training": train_df["label"].value_counts(),
    "Validation": val_df["label"].value_counts(),
    "Testing": test_df["label"].value_counts()
}).fillna(0).astype(int)

print(distribution.sort_index())

print("\nData splitting completed successfully!")
