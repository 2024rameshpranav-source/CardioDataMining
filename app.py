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
DATA_PATH = BASE_DIR / "cardio_cleaned.csv"

st.title("❤️ Cardiovascular Disease Analysis")
st.caption("Data Mining Mini Project | CA2")


# ---------------- LOAD SAVED MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------- LOAD CLEANED DATASET ----------------
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    metadata = load_metadata()
except FileNotFoundError as error:
    st.error(
        f"Required model file not found: {error.filename}. "
        "Ensure cardio_model.pkl and model_metadata.json "
        "are present beside app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load model files: {error}")
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
        "This project applies data mining classification techniques "
        "to a public cardiovascular disease dataset. It demonstrates "
        "data preprocessing, exploratory data analysis, model training, "
        "hyperparameter tuning, evaluation, and interactive prediction."
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
        "Explore the cleaned cardiovascular dataset. "
        "The CSV is loaded automatically from the project repository."
    )

    if not DATA_PATH.exists():
        st.error(
            "cardio_cleaned.csv was not found. "
            "Make sure the file is committed to GitHub in the same "
            "directory as app.py."
        )
        st.stop()

    try:
        data = load_dataset()
    except Exception as error:
        st.error(f"Could not read cardio_cleaned.csv: {error}")
        st.stop()

    required_columns = {
        "cardio",
        "age_years",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc"
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        st.error(
            "The cleaned dataset is missing these required columns: "
            + ", ".join(sorted(missing_columns))
        )
        st.stop()

    # Dataset summary
    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset records", f"{len(data):,}")
    c2.metric("Number of columns", f"{len(data.columns):,}")
    c3.metric("Missing values", f"{int(data.isnull().sum().sum()):,}")

    with st.expander("View dataset information"):
        summary = pd.DataFrame({
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).values,
            "Missing Values": data.isnull().sum().values,
            "Unique Values": data.nunique().values
        })
        st.dataframe(summary, use_container_width=True)

    # Target distribution
    st.subheader("Target Distribution")

    counts = data["cardio"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Cardiovascular Disease Class Distribution")
    ax.set_xlabel("Cardio (0 = No, 1 = Yes)")
    ax.set_ylabel("Number of Records")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(value) for value in counts.index], rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Age distribution
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
    ax.set_title("Age Distribution by Cardiovascular Disease Class")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Records")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Correlation heatmap
    st.subheader("Feature Correlation")

    numeric_data = data.select_dtypes(include="number")
    correlation = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False,
        ax=ax
    )
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Additional feature exploration
    st.subheader("Explore Individual Features")

    available_features = [
        "age_years",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc"
    ]

    selected_feature = st.selectbox(
        "Select a feature",
        available_features
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=data,
        x=selected_feature,
        bins=25,
        kde=True,
        ax=ax
    )
    ax.set_title(f"Distribution of {selected_feature}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Dataset preview
    with st.expander("View cleaned dataset preview"):
        st.dataframe(data.head(20), use_container_width=True)

    st.caption(
        "Visualizations use the cleaned dataset stored in the repository."
    )


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

    if "best_cv_f1" in metadata:
        st.metric(
            "Best Cross-Validation F1-score",
            f'{metadata["best_cv_f1"]:.4f}'
        )

    st.subheader("Best Hyperparameters")
    st.json(metadata["best_parameters"])

    st.write(
        "These metrics are loaded from the saved metadata generated "
        "during model development and refer to the held-out test set."
    )

    st.warning(
        "The model's performance is intended for an academic demonstration "
        "and does not establish clinical reliability."
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
            "Height (cm)",
            min_value=120.0,
            max_value=220.0,
            value=165.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=70.0
        )

        ap_hi = st.number_input(
            "Systolic blood pressure",
            min_value=70,
            max_value=250,
            value=120
        )

        ap_lo = st.number_input(
            "Diastolic blood pressure",
            min_value=40,
            max_value=200,
            value=80
        )

        cholesterol = st.selectbox(
            "Cholesterol level",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal (1)",
                2: "Above normal (2)",
                3: "Well above normal (3)"
            }[x]
        )

        gluc = st.selectbox(
            "Glucose level",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal (1)",
                2: "Above normal (2)",
                3: "Well above normal (3)"
            }[x]
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

            missing_features = [
                feature for feature in feature_order
                if feature not in input_values
            ]

            if missing_features:
                st.error(
                    "Prediction cannot proceed because the form is missing "
                    "model features: " + ", ".join(missing_features)
                )
            else:
                input_df = pd.DataFrame(
                    [[input_values[col] for col in feature_order]],
                    columns=feature_order
                )

                try:
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

                except Exception as error:
                    st.error(f"Prediction failed: {error}")
