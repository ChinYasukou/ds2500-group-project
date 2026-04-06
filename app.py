import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from Graph_with_KNN import (
    load_clean_data,
    prepare_graph_data,
    plot_bmi_by_income,
    plot_bmi_by_education,
    plot_general_health_by_income,
    plot_general_health_by_education,
    plot_diabetes_by_income,
    plot_diabetes_by_education,
    plot_hypertension_by_income,
    plot_cholesterol_by_income,
    plot_age_vs_bmi,
    plot_bmi_by_sex
)


# =========================
# page setup
# =========================
st.set_page_config(page_title="Health Prediction System", layout="wide")

st.title("Health Prediction System")
st.write("Predict health risks based on socioeconomic factors.")


# =========================
# load data
# =========================
@st.cache_data
def load_project_data():
    return load_clean_data()


@st.cache_data
def load_graph_data():
    df = load_clean_data()
    return prepare_graph_data(df)


df = load_project_data()
graph_df = load_graph_data()


# =========================
# train models
# =========================
@st.cache_resource
def train_models(dataframe):
    features = ["income", "education", "age", "sex"]
    targets = ["diabetes", "hypertension", "cholesterol"]

    df_model = dataframe.copy()
    df_model = df_model.dropna(subset=features + targets)

    X = df_model[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {}
    target_summaries = {}

    for target in targets:
        y = df_model[target]

        model = KNeighborsClassifier(n_neighbors=15)
        model.fit(X_scaled, y)

        models[target] = model
        target_summaries[target] = y.value_counts().to_dict()

    return scaler, models, target_summaries


scaler, models, target_summaries = train_models(df)


# =========================
# sidebar input
# =========================
st.sidebar.header("Enter Your Information")

income = st.sidebar.slider("Income Level", 1, 11, 6)
education = st.sidebar.slider("Education Level", 1, 6, 4)
age = st.sidebar.slider("Age", 18, 80, 30)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])

sex_value = 1 if sex == "Male" else 0

user_input = pd.DataFrame([{
    "income": income,
    "education": education,
    "age": age,
    "sex": sex_value
}])

user_input_scaled = scaler.transform(user_input)


# =========================
# prediction helpers
# =========================
def predict_risk(model, scaled_input):
    prediction = model.predict(scaled_input)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_input)[0]
        classes = model.classes_
        prob_map = dict(zip(classes, probabilities))
        risk_score = prob_map.get(1, 0.0)
    else:
        risk_score = 1.0 if prediction == 1 else 0.0

    return prediction, risk_score


def risk_label(prediction):
    return "High" if prediction == 1 else "Low"


def risk_color_box(title, prediction, score):
    label = risk_label(prediction)
    percent = round(score * 100, 1)

    if prediction == 1:
        st.error(f"{title}: {label} ({percent}%)")
    else:
        st.success(f"{title}: {label} ({percent}%)")


# =========================
# dataset preview
# =========================
st.subheader("Dataset Preview")
st.dataframe(df.head())


# =========================
# model info
# =========================
with st.expander("Model Information"):
    st.write("Features used for prediction: income, education, age, and sex.")
    st.write("Model used: K-Nearest Neighbors (KNN) with standardized inputs.")
    st.write("Target distribution:")
    st.write(target_summaries)


# =========================
# predictions
# =========================
pred_diabetes, score_diabetes = predict_risk(models["diabetes"], user_input_scaled)
pred_hypertension, score_hypertension = predict_risk(models["hypertension"], user_input_scaled)
pred_cholesterol, score_cholesterol = predict_risk(models["cholesterol"], user_input_scaled)

st.subheader("Prediction Results")

col1, col2, col3 = st.columns(3)

with col1:
    risk_color_box("Diabetes Risk", pred_diabetes, score_diabetes)

with col2:
    risk_color_box("Hypertension Risk", pred_hypertension, score_hypertension)

with col3:
    risk_color_box("Cholesterol Risk", pred_cholesterol, score_cholesterol)


# =========================
# quick interpretation
# =========================
st.subheader("Quick Interpretation")

messages = []

if pred_diabetes == 1:
    messages.append("The model predicts a higher diabetes risk for this input profile.")
else:
    messages.append("The model predicts a lower diabetes risk for this input profile.")

if pred_hypertension == 1:
    messages.append("The model predicts a higher hypertension risk for this input profile.")
else:
    messages.append("The model predicts a lower hypertension risk for this input profile.")

if pred_cholesterol == 1:
    messages.append("The model predicts a higher cholesterol risk for this input profile.")
else:
    messages.append("The model predicts a lower cholesterol risk for this input profile.")

for message in messages:
    st.write(f"- {message}")


# =========================
# visualizations
# =========================
st.subheader("Data Visualizations")

tab1, tab2, tab3 = st.tabs(["BMI and Health", "Disease Rates", "Demographics"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_bmi_by_income(graph_df))
        st.pyplot(plot_general_health_by_income(graph_df))
        st.pyplot(plot_age_vs_bmi(graph_df))
    with col2:
        st.pyplot(plot_bmi_by_education(graph_df))
        st.pyplot(plot_general_health_by_education(graph_df))
        st.pyplot(plot_bmi_by_sex(graph_df))

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_diabetes_by_income(graph_df))
        st.pyplot(plot_hypertension_by_income(graph_df))
    with col2:
        st.pyplot(plot_diabetes_by_education(graph_df))
        st.pyplot(plot_cholesterol_by_income(graph_df))

with tab3:
    st.write("These graphs help show how socioeconomic factors and demographics relate to BMI and chronic disease outcomes in the dataset.")


# =========================
# footer
# =========================
st.write("---")
st.write("DS2500 Final Project")