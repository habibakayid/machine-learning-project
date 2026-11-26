import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1150px;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .hero {
        background: linear-gradient(120deg, #0f766e 0%, #2563eb 100%);
        border-radius: 20px;
        padding: 28px 34px;
        color: #ffffff;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
    }
    .hero h1 {color: #ffffff; margin: 0; font-size: 2.1rem; padding: 0;}
    .hero p  {color: #e0f2fe; margin: 6px 0 0 0; font-size: 1.05rem;}

    .section-title {
        font-size: 1.1rem; font-weight: 700; margin-bottom: .4rem;
    }

    .result-card {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        border-radius: 20px;
        padding: 30px 24px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(5, 150, 105, 0.30);
        margin-bottom: 1rem;
    }
    .result-card .label {font-size: .95rem; opacity: .9; letter-spacing: .5px;}
    .result-card .amount {font-size: 2.9rem; font-weight: 800; line-height: 1.2;}
    .result-card .sub {font-size: .85rem; opacity: .85; margin-top: 4px;}

    .placeholder-card {
        border: 2px dashed rgba(100, 116, 139, 0.5);
        border-radius: 20px;
        padding: 60px 24px;
        text-align: center;
        opacity: .75;
    }

    .chip {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-weight: 600; font-size: .85rem; color: #ffffff;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(120deg, #0f766e, #2563eb);
        color: #ffffff; border: none; border-radius: 12px;
        padding: .6rem 1.2rem; font-weight: 700; width: 100%;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.1); color: #ffffff;
    }
    div[data-testid="stForm"] {border-radius: 18px; padding: 1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏥 Medical Insurance Charges Predictor</h1>
        <p>Enter the patient's information and get an instant estimate of the yearly insurance charges.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


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
    st.markdown("### ⚙️ Settings")
    model_name = st.selectbox("Choose model", list(available_models.keys()))
    st.caption(
        "The Voting Regressor averages Linear Regression and KNN predictions, "
        "while Linear Regression is the simpler baseline."
    )
    st.divider()
    st.caption("Built with Streamlit and scikit-learn.")

artifacts = load_artifacts(available_models[model_name])
preprocessor = artifacts["preprocessor"]
model = artifacts["model"]


def predict_charges(patient: pd.DataFrame) -> float:
    processed = preprocessor.transform(patient)
    return max(float(model.predict(processed)[0]), 0.0)


def bmi_category(value: float):
    if value < 18.5:
        return "Underweight", "#2563eb"
    if value < 25:
        return "Normal", "#059669"
    if value < 30:
        return "Overweight", "#d97706"
    return "Obese", "#dc2626"



left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="section-title">👤 Patient information</div>', unsafe_allow_html=True)
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

        submitted = st.form_submit_button("🔮 Predict charges")

with right:
    st.markdown('<div class="section-title">💰 Estimated charges</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown(
            '<div class="placeholder-card">Fill in the form and press <b>Predict charges</b><br>'
            "to see the estimate here.</div>",
            unsafe_allow_html=True,
        )
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
        category, colour = bmi_category(bmi)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">PREDICTED YEARLY CHARGES</div>
                <div class="amount">${prediction:,.2f}</div>
                <div class="sub">Model: {model_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric("Monthly (approx.)", f"${prediction / 12:,.0f}")
        m2.markdown(
            f'BMI category<br><span class="chip" style="background:{colour}">{category}</span>',
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Extra insight: how smoking changes the estimate
# ----------------------------------------------------------------------------
if submitted:
    st.divider()
    st.markdown('<div class="section-title">📊 What-if: effect of smoking</div>', unsafe_allow_html=True)

    smoker_no = predict_charges(patient.assign(smoker="no"))
    smoker_yes = predict_charges(patient.assign(smoker="yes"))

    chart_df = pd.DataFrame(
        {"Predicted charges ($)": [smoker_no, smoker_yes]},
        index=["Non-smoker", "Smoker"],
    )

    w1, w2 = st.columns([1, 1.4], gap="large")
    with w1:
        st.metric("Non-smoker", f"${smoker_no:,.0f}")
        st.metric(
            "Smoker",
            f"${smoker_yes:,.0f}",
            delta=f"{smoker_yes - smoker_no:+,.0f} vs non-smoker",
            delta_color="inverse",
        )
    with w2:
        st.bar_chart(chart_df)

    with st.expander("Input summary"):
        st.dataframe(patient, hide_index=True)
