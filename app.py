import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥")
st.title("🏥 Medical Insurance Charges Predictor")
st.write("Enter the patient's information to predict the insurance charges.")

BASE_DIR = Path(__file__).parent
LR_PATH = BASE_DIR / "insurance_model.pkl"
VOTING_PATH = BASE_DIR / "insurance_voting_model.pkl"


@st.cache_resource
def load_artifacts(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


available_models = {}
if VOTING_PATH.exists():
    available_models["Voting Regressor (Linear Regression + KNN) - Best"] = VOTING_PATH
if LR_PATH.exists():
    available_models["Linear Regression"] = LR_PATH

if not available_models:
    st.error(
        "No saved model found. Make sure insurance_model.pkl and "
        "insurance_voting_model.pkl are in the repository."
    )
    st.stop()

model_name = st.sidebar.selectbox("Choose model", list(available_models.keys()))
artifacts = load_artifacts(available_models[model_name])
preprocessor = artifacts["preprocessor"]
model = artifacts["model"]

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)

    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    submitted = st.form_submit_button("Predict")

if submitted:
    new_patient = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )

    processed = preprocessor.transform(new_patient)
    prediction = max(float(model.predict(processed)[0]), 0.0)

    st.success(f"Predicted insurance charges: **${prediction:,.2f}**")
    st.caption(f"Model used: {model_name}")

    with st.expander("Input summary"):
        st.dataframe(new_patient)
