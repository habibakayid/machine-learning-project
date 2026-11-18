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


@st.cache_resource
def load_model(path, modified_ns):
    return joblib.load(path)


def show_prediction(model_name, source, caption):
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        st.error("This image could not be opened. Upload a valid JPEG, PNG, or WebP file.")
        return
    path = MODELS / FILES[model_name]
    try:
        with st.spinner("Classifying flower…"):
            bundle = load_model(path, path.stat().st_mtime_ns)
            label, probabilities, cluster = predict_image(bundle, image)
    except FileNotFoundError:
        st.error("The saved model is missing. Run the training command in the project README.")
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
    model_name = st.selectbox("Model", list(FILES))
    uploaded = st.file_uploader("Flower image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None and st.button("Try a dataset example"):
        st.session_state.use_example = True

with result:
    if uploaded is not None:
        show_prediction(model_name, uploaded, uploaded.name)
    elif st.session_state.get("use_example"):
        show_prediction(model_name, EXAMPLE, "Daisy example from the project dataset")
    else:
        with st.container(border=True):
            st.subheader("Classify a flower")
            st.write("Upload a flower photo or try the dataset example.")
