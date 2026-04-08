from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")

MODEL_PATH = ARTIFACTS_DIR / "sklearn_model.joblib"
TRAIN_PATH = DATA_DIR / "train.csv"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="ML Frameworks Demo",
    page_icon="📊",
    layout="centered",
)

st.title("ML Frameworks Demo")
st.caption("Adult Income prediction using a trained scikit-learn pipeline.")

st.write(
    "Enter a person's details and the app will predict whether the income is likely to be **<=50K** or **>50K**."
)

# -------------------------------
# Load data and model
# -------------------------------
@st.cache_data
def load_training_data():
    return pd.read_csv(TRAIN_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_comparison_metrics():
    if COMPARISON_PATH.exists():
        return pd.read_json(COMPARISON_PATH, typ="series")
    return None


train_df = load_training_data()
model = load_model()
comparison_metrics = load_comparison_metrics()

# -------------------------------
# Input options from training data
# -------------------------------
feature_cols = [col for col in train_df.columns if col != "target"]

categorical_cols = train_df[feature_cols].select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = train_df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()

st.subheader("Enter person details")

with st.form("prediction_form"):
    user_input = {}

    col1, col2 = st.columns(2)

    with col1:
        for col in numeric_cols[: len(numeric_cols) // 2]:
            min_val = int(train_df[col].min())
            max_val = int(train_df[col].max())
            default_val = int(train_df[col].median())
            user_input[col] = st.number_input(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
            )

    with col2:
        for col in numeric_cols[len(numeric_cols) // 2 :]:
            min_val = int(train_df[col].min())
            max_val = int(train_df[col].max())
            default_val = int(train_df[col].median())
            user_input[col] = st.number_input(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
            )

    for col in categorical_cols:
        options = sorted(train_df[col].dropna().astype(str).unique().tolist())
        user_input[col] = st.selectbox(col, options)

    submitted = st.form_submit_button("Predict")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction Result")

    if prediction == ">50K":
        st.success(f"Predicted income class: {prediction}")
    else:
        st.info(f"Predicted income class: {prediction}")

    st.write(f"Probability of **>50K**: **{probability:.2%}**")

    st.markdown("### What this means")
    st.write(
        "This is a demo prediction based on patterns learned from the Adult Income dataset. "
        "It is not a real-world financial assessment."
    )

# -------------------------------
# Model summary
# -------------------------------
st.markdown("### Model Summary")
st.write(
    "The app uses the **scikit-learn** model because it is the simplest to load and deploy for a small demo."
)

if comparison_metrics is not None:
    st.markdown("### Framework Comparison")
    comparison_df = pd.DataFrame(comparison_metrics).T
    st.dataframe(comparison_df, use_container_width=True)
else:
    st.caption("Comparison metrics file not found yet.")