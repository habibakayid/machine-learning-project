"""Streamlit inference for the five-class Flowers project."""

import sys
from pathlib import Path

import joblib
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part_b_images" / "rebuild"))
from dataset_preparation import CLASSES, RAW
from train_models import FILES, MODELS, predict_image

EXAMPLE = RAW / "daisy" / "34696729796_190b1dfdf1_n.jpg"
GENERAL_MODELS = MODELS.parent / "generalization" / "models"
FINAL_MODELS = MODELS.parent / "final_optimization"
MODEL_EXPLANATIONS = {
    "Logistic Regression": "Uses a nonlinear map of handcrafted features to estimate class probabilities.",
    "KNN Classifier": "Classifies a photo by a vote from similar training photos.",
    "K-Means": "Finds the nearest cluster and shows the flower label assigned from training photos.",
}


def model_path(model_name, scope):
    if "--archived-rebuild" in sys.argv:
        return MODELS / FILES[model_name]
    if model_name == "KNN Classifier" and "--archived-knn" in sys.argv:
        return GENERAL_MODELS / FILES[model_name]
    if model_name == "K-Means" and "--archived-kmeans" in sys.argv:
        return GENERAL_MODELS / FILES[model_name]
    folder = model_name.replace(" ", "_").lower()
    return FINAL_MODELS / scope / folder / "selected.joblib"


@st.cache_resource
def load_model(path, modified_ns):
    return joblib.load(path)


def show_prediction(model_name, scope, source, caption):
    try:
        with Image.open(source) as opened:
            if opened.format not in ("JPEG", "PNG", "WEBP"):
                st.error("This image format is unsupported. Upload a JPEG, PNG, or WebP file.")
                return
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        st.error("This image could not be opened. Upload a valid JPEG, PNG, or WebP file.")
        return
    path = model_path(model_name, scope)
    try:
        with st.spinner("Classifying flower…"):
            bundle = load_model(path, path.stat().st_mtime_ns)
            label, probabilities, cluster = predict_image(bundle, image)
    except FileNotFoundError:
        st.error("The saved model is missing. Restore the model file listed in the README.")
        return
    except (OSError, ValueError, ImportError) as error:
        st.error(f"The model could not classify this image. {error}")
        return

    st.caption(f"Prediction · {model_name}")
    st.header(label.title())
    preview = image.copy()
    preview.thumbnail((720, 420))
    st.image(preview, caption=caption, width="stretch")
    if probabilities is None:
        st.info(f"Cluster {cluster + 1} was assigned this label using the training images.")
        return
    st.markdown("**Estimated class probabilities**")
    for name in CLASSES:
        probability = float(probabilities[name])
        st.progress(probability, text=f"{name.title()}  ·  {probability:.1%}")


st.set_page_config(page_title="Flowers Classification", page_icon="🌼", layout="wide")
st.html("""
<style>
.stMainBlockContainer {max-width: 66rem; padding-top: 3.25rem;}
.stMain h1, .stMain h2 {font-family: "Instrument Serif", Georgia, serif; font-weight: 500;}
.stMain h1 {font-size: clamp(2.15rem, 6vw, 3.65rem); line-height: 1.05;}
.stMain h2 {font-size: 2rem;}
[data-testid="stImage"] img {height: auto; max-height: 25rem; object-fit: contain;
  background: #e7ecde; border-radius: .5rem;}
[data-testid="stFileUploaderDropzone"] {min-height: 5.5rem;}
@media (max-width: 640px) {.stMainBlockContainer {padding-top: 2.25rem;}}
</style>
""")
st.title("Flowers Classification")
st.markdown("Daisy · Dandelion · Rose · Sunflower · Tulip")
st.divider()
controls, result = st.columns([2, 3], gap="medium")
with controls:
    experiment = st.selectbox(
        "Training data",
        ["Kaggle only (primary)", "Kaggle + field (supplemental)"],
        index=1 if "--generalization" in sys.argv else 0,
    )
    scope = "primary" if experiment.startswith("Kaggle only") else "supplemental"
    model_name = st.selectbox("Model", list(FILES))
    st.caption(MODEL_EXPLANATIONS[model_name])
    uploaded = st.file_uploader("Flower image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None and st.button("Try a dataset example"):
        st.session_state.use_example = True

with result:
    if uploaded is not None:
        show_prediction(model_name, scope, uploaded, uploaded.name)
    elif st.session_state.get("use_example"):
        show_prediction(model_name, scope, EXAMPLE, "Daisy example from the project dataset")
    else:
        with st.container(border=True):
            st.subheader("Classify a flower")
            st.write("Upload a flower photo or try the dataset example.")
