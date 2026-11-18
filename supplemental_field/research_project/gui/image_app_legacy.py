
from pathlib import Path
import json

import streamlit as st
import numpy as np
import pandas as pd
import joblib

from PIL import Image, ImageOps
from skimage.feature import hog
from skimage.color import rgb2hsv


# ==========================================
# 1. Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "models"

IMAGE_SIZE = (64, 64)

MODEL_FILES = {
    "Logistic Regression (Improved)": "part_b_logistic_hog_rgb_hsv.joblib",
    "KNN Classifier": "part_b_knn_combined.joblib",
    "K-Means": "part_b_kmeans_combined.joblib"
}

st.set_page_config(
    page_title="Flowers Classification",
    page_icon="🌸",
    layout="centered"
)


# ==========================================
# 2. Load Saved Model
# ==========================================

@st.cache_resource
def load_model(model_name):

    model_path = MODELS_PATH / MODEL_FILES[model_name]

    model = joblib.load(model_path)

    return model


# ==========================================
# 3. Image Preprocessing
# ==========================================

def preprocess_image(image):

    image = ImageOps.exif_transpose(image).convert("RGB")

    image = ImageOps.pad(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    return image_array


# ==========================================
# 4. Extract HOG Features
# ==========================================

def extract_hog(image):

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        channel_axis=-1,
        feature_vector=True
    )

    return features.astype(np.float32)


# ==========================================
# 5. Extract RGB Color Histogram
# ==========================================

def extract_rgb(image):

    histograms = []

    for channel in range(3):

        hist, _ = np.histogram(
            image[:, :, channel],
            bins=16,
            range=(0.0, 1.0)
        )

        hist = hist.astype(np.float32)
        hist /= max(hist.sum(), 1.0)

        histograms.append(hist)

    return np.concatenate(histograms)


# ==========================================
# 6. Extract Spatial HSV Features
# ==========================================

def hsv_histogram(region):

    features = []

    for channel in range(3):

        hist, _ = np.histogram(
            region[:, :, channel],
            bins=16,
            range=(0.0, 1.0)
        )

        hist = hist.astype(np.float32)
        hist /= max(hist.sum(), 1.0)

        features.extend(hist)

    return np.array(features, dtype=np.float32)


def extract_spatial_hsv(image):

    hsv = rgb2hsv(image)

    # Global histogram
    global_features = hsv_histogram(hsv)

    # Divide image into four regions
    regions = []

    for row in np.array_split(hsv, 2, axis=0):
        for region in np.array_split(row, 2, axis=1):
            regions.append(region)

    spatial_features = [
        hsv_histogram(region)
        for region in regions
    ]

    return np.concatenate(
        [global_features] + spatial_features
    )


# ==========================================
# 7. Streamlit Interface
# ==========================================

st.title("🌸 Flowers Image Classification")

st.write(
    "Upload a flower image and let the trained "
    "Machine Learning model predict its class."
)

st.divider()

model_name = st.selectbox(
    "Select Machine Learning Model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.file_uploader(
    "Upload Flower Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 8. Prediction
# ==========================================

if uploaded_file is not None:

    try:

        original_image = Image.open(uploaded_file)

        st.image(
            original_image,
            caption="Uploaded Flower Image",
            width=330
        )

        if st.button(
            "Predict Flower",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing image..."):

                # Preprocess uploaded image
                image = preprocess_image(original_image)

                # Extract HOG features
                hog_features = extract_hog(image)

                # ==================================
                # Select Features Based on Model
                # ==================================

                if model_name == "Logistic Regression (Improved)":

                    rgb_features = extract_rgb(image)
                    hsv_features = extract_spatial_hsv(image)

                    features = np.concatenate([
                        hog_features,
                        rgb_features,
                        hsv_features
                    ]).reshape(1, -1)

                else:

                    rgb_features = extract_rgb(image)

                    features = np.concatenate([
                        hog_features,
                        rgb_features
                    ]).reshape(1, -1)

                # Display feature dimensions
                st.write("Input Shape:", features.shape)

                # Load selected model
                model = load_model(model_name)

                # ==================================
                # K-Means Prediction
                # ==================================

                if model_name == "K-Means":

                    cluster_id = int(
                        model.predict(features)[0]
                    )

                    mapping_path = (
                        MODELS_PATH
                        / "part_b_kmeans_combined_mapping.json"
                    )

                    with open(
                        mapping_path,
                        "r",
                        encoding="utf-8"
                    ) as file:

                        mapping = json.load(file)

                    prediction = mapping[str(cluster_id)]

                    probabilities = None

                # ==================================
                # Logistic Regression / KNN
                # ==================================

                else:

                    prediction = model.predict(features)[0]

                    probabilities = model.predict_proba(
                        features
                    )[0]

            # ======================================
            # 9. Display Prediction Results
            # ======================================

            st.divider()

            st.success(
                f"Predicted Flower: {prediction.capitalize()}"
            )

            st.write(f"Model Used: {model_name}")

            # Supervised model probabilities
            if probabilities is not None:

                confidence = probabilities.max() * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                probability_df = pd.DataFrame({
                    "Flower": model.classes_,
                    "Probability": probabilities
                })

                probability_df = probability_df.set_index(
                    "Flower"
                )

                st.subheader("Class Probabilities")

                st.bar_chart(probability_df)

                st.caption(
                    "Probabilities are model estimates, "
                    "not guaranteed accuracy."
                )

            # K-Means cluster information
            else:

                st.write(
                    f"Assigned Cluster: {cluster_id}"
                )

                st.info(
                    "K-Means is an unsupervised model. "
                    "Its cluster was mapped to a flower label."
                )

    except Exception as error:

        st.error(f"Prediction Error: {error}")


# ==========================================
# 10. Footer
# ==========================================

st.divider()

st.caption(
    "Machine Learning Project | Part B - Image Classification"
)
