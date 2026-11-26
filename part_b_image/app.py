import pandas as pd
import streamlit as st

from predict_flower import MODEL_FILES, load_models, predict_all

st.set_page_config(page_title="Flower Recognition", layout="wide")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "kmeans": "K-Means",
}

# Test accuracy reported in the project README
TEST_ACCURACY = {
    "logistic_regression": "78.17%",
    "knn": "71.16%",
    "kmeans": "57.95%",
}


@st.cache_resource
def get_models():
    return load_models()


st.title("Flower Recognition")
st.write(
    "Upload a photo of a flower and the trained models will classify it as "
    "daisy, sunflower or tulip."
)

with st.sidebar:
    st.header("Settings")
    model_key = st.selectbox(
        "Choose model",
        list(MODEL_FILES.keys()),
        format_func=lambda key: MODEL_LABELS[key],
    )
    st.caption(
        "Logistic Regression is the best model on the test set. "
        "K-Means is unsupervised, so it is the least accurate."
    )

models = get_models()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Image")
    uploaded_file = st.file_uploader(
        "Upload a flower image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image")

with right:
    st.subheader("Result")
    if uploaded_file is None:
        with st.container(border=True):
            st.write("Upload an image to see the prediction here.")
    else:
        try:
            all_predictions = predict_all(uploaded_file, models)
        except Exception as error:
            st.error(f"Could not read this image: {error}")
            st.stop()

        with st.container(border=True):
            st.metric(
                f"Prediction ({MODEL_LABELS[model_key]})",
                all_predictions[model_key].title(),
            )
            st.write(f"Test accuracy of this model: **{TEST_ACCURACY[model_key]}**")

if uploaded_file is not None:
    st.divider()
    st.subheader("All models on this image")

    table = pd.DataFrame(
        {
            "Model": [MODEL_LABELS[key] for key in MODEL_FILES],
            "Prediction": [all_predictions[key].title() for key in MODEL_FILES],
            "Test accuracy": [TEST_ACCURACY[key] for key in MODEL_FILES],
        }
    )
    st.dataframe(table, hide_index=True)

    if len(set(all_predictions.values())) > 1:
        st.write("The models do not agree on this image.")
    else:
        st.write("All three models agree.")