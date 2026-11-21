# Flowers Recognition — GUI Integration

This project recognizes **daisy**, **sunflower**, and **tulip** using three trained models: Logistic Regression, KNN, and K-Means.

## Folder structure

```text
Part_B_Image/
├── flowers_recognition.ipynb
├── models/
│   ├── logistic_regression.joblib
│   ├── knn.joblib
│   └── kmeans.joblib
├── predict_flower.py
├── requirements.txt
└── README.md
```

The notebook documents model training and evaluation; the GUI does **not** need to rerun it. The flower dataset is also **not needed** to predict new images.

## 1. Setup

Use Python **3.12**. Open a terminal inside `Part_B_Image` and run:

```powershell
py -3.12 -m pip install -r requirements.txt
```

If using VS Code, select the same Python 3.12 interpreter for the project.

## 2. Test the three models

Put any flower image on your computer and run:

```powershell
py -3.12 predict_flower.py "C:\path\to\flower.jpg"
```

Each model should print one of the three class names (`daisy`, `sunflower`, `tulip`). These are predictions, not guaranteed correct labels.

## 3. Connect to the GUI

```python
from predict_flower import load_models, predict_flower, predict_all

models = load_models()  # Load once when the GUI starts

# Predict with one model:
flower = predict_flower("flower.jpg", "logistic_regression", models)
print(flower)

# Or compare predictions from all three models:
results = predict_all("flower.jpg", models)
print(results)
```

Model keys: `logistic_regression`, `knn`, `kmeans`.

For **Streamlit**, the uploaded image file can be passed directly to `predict_flower(uploaded_file, "knn", models)`. For **Tkinter**, pass the image path from the file picker. The input may also be a PIL Image.

## 4. Important implementation notes

- The GUI must use `predict_flower.py` to apply exactly the **same resize, normalization, HOG and HSV color histogram extraction** as during training.
- Logistic Regression uses HOG and color features transformed with saved scalers and PCAs.
- KNN uses color histogram features.
- K-Means uses raw pixels transformed with its saved scaler and PCA. The saved `cluster_mapping` converts the predicted cluster ID to a flower class.
- The saved K-Means model here is the original **24-cluster** version, not the experimental 3–10 silhouette-selection version.
- A prediction is returned as a class label. The script does **not** provide calibrated confidence scores.
- Load `.joblib` files only from trusted sources. For compatibility, the files were trained with scikit-learn **1.5.2**.
- This package includes the **integration files only**. Copy them next to the existing `models` folder and notebook; do not put them inside the `models` folder.
