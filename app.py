
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="wide")

st.title("Medical Insurance Charges Predictor")
st.write("Enter the patient's information to estimate the yearly insurance charges.")


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

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Choose model", list(available_models.keys()))
    st.caption(
        "The Voting Regressor averages the predictions of Linear Regression "
        "and KNN. Linear Regression is the simpler baseline."
    )

artifacts = load_artifacts(available_models[model_name])
preprocessor = artifacts["preprocessor"]
model = artifacts["model"]


def predict_charges(patient: pd.DataFrame) -> float:
    processed = preprocessor.transform(patient)
    return max(float(model.predict(processed)[0]), 0.0)


def bmi_category(value: float) -> str:
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal"
    if value < 30:
        return "Overweight"
    return "Obese"



left, right = st.columns(2, gap="large")

with left:
    st.subheader("Patient information")
    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
            children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        with c2:
            sex = st.selectbox("Sex", ["male", "female"])
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

        submitted = st.form_submit_button("Predict")

with right:
    st.subheader("Result")

    if not submitted:
        with st.container(border=True):
            st.write("Fill in the form and press Predict to see the estimate here.")
    else:
        patient = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region],
            }
        )
        prediction = predict_charges(patient)

        with st.container(border=True):
            st.metric("Predicted yearly charges", f"${prediction:,.2f}")
            st.metric("Approx. monthly", f"${prediction / 12:,.2f}")
            st.write(f"BMI category: **{bmi_category(bmi)}**")
        st.caption(f"Model used: {model_name}")


if submitted:
    st.divider()
    st.subheader("Effect of smoking on the estimate")
    st.write("Same patient, same model, only the smoker value is changed.")

    smoker_no = predict_charges(patient.assign(smoker="no"))
    smoker_yes = predict_charges(patient.assign(smoker="yes"))

    comparison = pd.DataFrame(
        {
            "Smoker": ["no", "yes"],
            "Predicted charges": [f"${smoker_no:,.2f}", f"${smoker_yes:,.2f}"],
        }
    )
    st.dataframe(comparison, hide_index=True)
    st.write(f"Difference: ${smoker_yes - smoker_no:+,.2f} per year")

    with st.expander("Input summary"):
        st.dataframe(patient, hide_index=True)
