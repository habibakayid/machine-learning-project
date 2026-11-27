import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SMS Spam Detector", layout="wide")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

LANGUAGES = {"English": "english", "Arabic": "arabic"}
MODEL_LABELS = {
    "lr": "Logistic Regression",
    "svm": "Linear SVM",
}
MODEL_FILES = {
    "vectorizer": "tfidf_vectorizer.pkl",
    "lr": "logistic_regression_model.pkl",
    "svm": "linear_svm_model.pkl",
}
EXAMPLES = {
    "English": {
        "Spam example": "Congratulations! You have won a free prize. Call now to claim your reward.",
        "Normal example": "Hey, are we still meeting for lunch tomorrow at 1?",
    },
    "Arabic": {
        "Spam example": "اربح 10000 جنيه مجانا! اتصل الآن على 09012345678 لاستلام جائزتك",
        "Normal example": "هل سنلتقي غدا في الساعة الخامسة؟",
    },
}



@st.cache_resource
def get_english_stop_words():
    import nltk
    from nltk.corpus import stopwords

    try:
        words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english"))

    words_to_keep = {"not", "no", "nor", "what", "where", "how", "you", "doing"}
    return words - words_to_keep


def clean_english(text: str) -> str:
    stop_words = get_english_stop_words()
    text = text.lower()

    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"n't\b", " not", text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)


def clean_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|top|info)\b",
        " URLTOKEN ",
        text,
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn" and char != "\u0674"
    )

    text = text.replace("ـ", "")
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ی", "ي")
    text = text.replace("ک", "ك")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


CLEANERS = {"English": clean_english, "Arabic": clean_arabic}



@st.cache_resource
def load_models(language_folder: str):
    folder = MODELS_DIR / language_folder
    return {key: joblib.load(folder / filename) for key, filename in MODEL_FILES.items()}


def missing_files(language_folder: str):
    folder = MODELS_DIR / language_folder
    return [
        str((folder / filename).relative_to(BASE_DIR))
        for filename in MODEL_FILES.values()
        if not (folder / filename).is_file()
    ]


def is_spam(label) -> bool:
    return str(label).strip().lower() in {"spam", "1"}


def spam_probability(model, vector):
    """Logistic Regression only: probability of the spam class."""
    for index, cls in enumerate(model.classes_):
        if is_spam(cls):
            return float(model.predict_proba(vector)[0][index])
    return None



st.title("SMS Spam Detector")
st.write("Type or paste an SMS message to check whether it is spam.")

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Message language", list(LANGUAGES.keys()))
    model_key = st.selectbox(
        "Choose model",
        list(MODEL_LABELS.keys()),
        format_func=lambda key: MODEL_LABELS[key],
    )
    st.caption(
        "Both models use TF-IDF features. Logistic Regression can also give "
        "a spam probability."
    )

language_folder = LANGUAGES[language]
missing = missing_files(language_folder)
if missing:
    st.error(
        "Some model files are missing. Upload these files to the repository:\n\n"
        + "\n".join(f"- {path}" for path in missing)
    )
    st.stop()

models = load_models(language_folder)

if language == "Arabic":
    st.markdown(
        "<style>textarea {direction: rtl; text-align: right;}</style>",
        unsafe_allow_html=True,
    )

# Keep a separate text box (and state) for each language
text_key = f"message_{language_folder}"
if text_key not in st.session_state:
    st.session_state[text_key] = ""


def set_example(value: str):
    st.session_state[text_key] = value


left, right = st.columns(2, gap="large")

with left:
    st.subheader("Message")
    st.text_area("SMS text", key=text_key, height=180)

    example_cols = st.columns(len(EXAMPLES[language]))
    for col, (name, value) in zip(example_cols, EXAMPLES[language].items()):
        col.button(name, on_click=set_example, args=(value,), key=f"{text_key}_{name}")

    check = st.button("Check message", type="primary")

with right:
    st.subheader("Result")
    message = st.session_state[text_key].strip()

    if not check:
        with st.container(border=True):
            st.write("Enter a message and press Check message to see the result here.")
    elif not message:
        st.warning("Please enter a message first.")
    else:
        cleaned = CLEANERS[language](message)
        if not cleaned.strip():
            st.warning(
                "Nothing is left of this message after cleaning "
                "(for example, it only has stop words), so it cannot be classified."
            )
        else:
            vector = models["vectorizer"].transform([cleaned])
            predictions = {
                key: models[key].predict(vector)[0] for key in MODEL_LABELS
            }
            chosen = predictions[model_key]

            with st.container(border=True):
                st.metric(
                    f"Prediction ({MODEL_LABELS[model_key]})",
                    "Spam" if is_spam(chosen) else "Not spam (ham)",
                )
                probability = spam_probability(models["lr"], vector)
                if probability is not None:
                    st.write(
                        f"Spam probability (Logistic Regression): **{probability:.1%}**"
                    )

            table = pd.DataFrame(
                {
                    "Model": [MODEL_LABELS[key] for key in MODEL_LABELS],
                    "Prediction": [
                        "Spam" if is_spam(predictions[key]) else "Not spam (ham)"
                        for key in MODEL_LABELS
                    ],
                }
            )
            st.dataframe(table, hide_index=True)

            if len({is_spam(p) for p in predictions.values()}) > 1:
                st.write("The two models do not agree on this message.")

            with st.expander("Cleaned text used by the model"):
                st.write(cleaned)