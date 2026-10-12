
import json
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cardio Data Mining",
    page_icon="❤️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "cardio_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.title("❤️ Cardiovascular Disease Analysis")
st.caption("Data Mining Mini Project | CA2")

# ---------------- LOAD SAVED MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)

try:
    model = load_model()
    metadata = load_metadata()
except FileNotFoundError:
    st.error(
        "Model files not found. Place cardio_model.pkl "
        "and model_metadata.json beside app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Analysis",
        "Model Evaluation",
        "Prediction"
    ]
)

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.header("Project Overview")

    st.write(
        "This project applies classification algorithms to "
        "a public cardiovascular disease dataset. It demonstrates "
        "data preprocessing, exploratory analysis, model evaluation "
        "and prediction using a trained Random Forest classifier."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Original records", "70,000")
    c2.metric("Cleaned records", "67,943")
    c3.metric("Selected model", metadata["model_name"])

    st.subheader("Data Mining Workflow")

    st.markdown(
        """
        1. Dataset collection
        2. Data preprocessing and quality checks
        3. Exploratory data analysis
        4. Classification using Decision Tree and Random Forest
        5. Hyperparameter tuning and model evaluation
        6. Interactive prediction using the selected model
        """
    )

    st.info(
        "Educational project only. Predictions are not medical diagnoses."
    )

# ---------------- DATA ANALYSIS ----------------
elif page == "Data Analysis":
    st.header("Exploratory Data Analysis")

    st.write(
        "Upload the cleaned CSV used during model training "
        "to explore its distributions and relationships."
    )

    uploaded_csv = st.file_uploader(
        "Upload cleaned dataset CSV",
        type=["csv"]
    )

    if uploaded_csv is not None:
        data = pd.read_csv(uploaded_csv)

        required_columns = {
            "cardio", "age_years", "height", "weight",
            "ap_hi", "ap_lo", "cholesterol", "gluc"
        }

        if not required_columns.issubset(data.columns):
            st.error(
                "The CSV is missing required columns. "
                "Upload the cleaned dataset used for this project."
            )
        else:
            st.metric("Dataset records", f"{len(data):,}")

            st.subheader("Target Distribution")
            counts = data["cardio"].value_counts().sort_index()

            fig, ax = plt.subplots(figsize=(7, 4))
            counts.plot(kind="bar", ax=ax)
            ax.set_title("Cardiovascular Disease Class Distribution")
            ax.set_xlabel("Cardio (0 = No, 1 = Yes)")
            ax.set_ylabel("Records")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(
                data=data,
                x="age_years",
                hue="cardio",
                bins=25,
                element="step",
                ax=ax
            )
            ax.set_title("Age Distribution by Target Class")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Feature Correlation")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                data.corr(numeric_only=True),
                cmap="coolwarm",
                center=0,
                ax=ax
            )
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("View dataset preview"):
                st.dataframe(data.head(20), use_container_width=True)

# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.header("Model Performance")

    metrics = metadata["test_metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f'{metrics["accuracy"]:.2%}')
    c2.metric("Precision", f'{metrics["precision"]:.2%}')
    c3.metric("Recall", f'{metrics["recall"]:.2%}')
    c4.metric("F1-score", f'{metrics["f1_score"]:.2%}')

    st.subheader("Selected Model")
    st.write(metadata["model_name"])

    st.subheader("Best Hyperparameters")
    st.json(metadata["best_parameters"])

    st.write(
        "These metrics come from the held-out test evaluation "
        "recorded during model development."
    )

# ---------------- PREDICTION ----------------
elif page == "Prediction":
    st.header("Cardiovascular Disease Class Prediction")

    st.warning(
        "For educational demonstration only. This model is not "
        "validated for clinical decisions."
    )

    with st.form("prediction_form"):
        age_years = st.slider("Age (years)", 29, 65, 45)
        gender = st.selectbox(
            "Gender code",
            options=[1, 2],
            format_func=lambda x: (
                "Female (1)" if x == 1 else "Male (2)"
            )
        )

        height = st.number_input(
            "Height (cm)", 120.0, 220.0, 165.0
        )
        weight = st.number_input(
            "Weight (kg)", 30.0, 200.0, 70.0
        )
        ap_hi = st.number_input(
            "Systolic blood pressure",
            min_value=70, max_value=250, value=120
        )
        ap_lo = st.number_input(
            "Diastolic blood pressure",
            min_value=40, max_value=200, value=80
        )
        cholesterol = st.selectbox(
            "Cholesterol level", [1, 2, 3]
        )
        gluc = st.selectbox(
            "Glucose level", [1, 2, 3]
        )
        smoke = st.selectbox("Smokes", [0, 1])
        alco = st.selectbox("Consumes alcohol", [0, 1])
        active = st.selectbox("Physically active", [0, 1])

        submitted = st.form_submit_button("Predict class")

    if submitted:
        if ap_hi < ap_lo:
            st.error(
                "Systolic blood pressure should not be lower "
                "than diastolic pressure for this demonstration."
            )
        else:
            # Follow the exact feature order used during training
            input_values = {
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active,
                "age_years": float(age_years)
            }

            feature_order = metadata["features"]

            input_df = pd.DataFrame(
                [[input_values[col] for col in feature_order]],
                columns=feature_order
            )

            prediction = int(model.predict(input_df)[0])

            if prediction == 1:
                st.error(
                    "Model output: Cardiovascular disease class (1)"
                )
            else:
                st.success(
                    "Model output: No cardiovascular disease class (0)"
                )

            st.caption(
                "This is a model classification, not a medical diagnosis."
            )
