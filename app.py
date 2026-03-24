import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Health Data Dashboard", layout="wide")

st.title("Socioeconomic Factors and Health Outcomes")
st.write("Click the button below to generate the current project graphs.")

# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("brfss_survey_data_2024.csv")

    cols = ["_BMI5", "INCOME3", "EDUCA", "GENHLTH", "DIABETE4"]
    df = df[cols].copy()
    df = df.dropna()

    df["_BMI5"] = df["_BMI5"] / 100
    df = df[df["EDUCA"].isin([1, 2, 3, 4, 5, 6, 9])]
    df = df[df["INCOME3"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 77, 99])]
    df = df[df["GENHLTH"].isin([1, 2, 3, 4, 5])]

    return df

df = load_data()

# =========================
# Prepare diabetes data
# =========================
df_diabetes = df[df["DIABETE4"].isin([1, 3])].copy()
df_diabetes["DIABETE4"] = df_diabetes["DIABETE4"].replace({1: 1, 3: 0})

# =========================
# Button to show graphs
# =========================
if st.button("Run Analysis", type="primary"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BMI by Income Level")
        fig, ax = plt.subplots(figsize=(7, 4))
        df.boxplot(column="_BMI5", by="INCOME3", ax=ax)
        ax.set_title("BMI by Income Level")
        ax.set_xlabel("Income Level")
        ax.set_ylabel("BMI")
        fig.suptitle("")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("BMI by Education Level")
        fig, ax = plt.subplots(figsize=(7, 4))
        df.boxplot(column="_BMI5", by="EDUCA", ax=ax)
        ax.set_title("BMI by Education Level")
        ax.set_xlabel("Education Level")
        ax.set_ylabel("BMI")
        fig.suptitle("")
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("General Health by Income")
        grouped = df.groupby("INCOME3")["GENHLTH"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        grouped.plot(kind="bar", ax=ax)
        ax.set_title("General Health by Income")
        ax.set_xlabel("Income Level")
        ax.set_ylabel("Average General Health Score")
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.subheader("General Health by Education")
        grouped = df.groupby("EDUCA")["GENHLTH"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        grouped.plot(kind="bar", ax=ax)
        ax.set_title("General Health by Education")
        ax.set_xlabel("Education Level")
        ax.set_ylabel("Average General Health Score")
        st.pyplot(fig)
        plt.close(fig)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Diabetes Rate by Income")
        grouped = df_diabetes.groupby("INCOME3")["DIABETE4"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        grouped.plot(kind="bar", ax=ax)
        ax.set_title("Diabetes Rate by Income")
        ax.set_xlabel("Income Level")
        ax.set_ylabel("Diabetes Rate")
        st.pyplot(fig)
        plt.close(fig)

    with col6:
        st.subheader("Diabetes Rate by Education")
        grouped = df_diabetes.groupby("EDUCA")["DIABETE4"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        grouped.plot(kind="bar", ax=ax)
        ax.set_title("Diabetes Rate by Education")
        ax.set_xlabel("Education Level")
        ax.set_ylabel("Diabetes Rate")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Key Insights")

    st.markdown("""
    - General health appears to improve as income level increases.
    - General health also appears to improve with higher education levels.
    - Diabetes rates generally decrease as income and education levels rise.
    - BMI distributions are broadly similar across groups, although some higher socioeconomic groups show slightly lower median BMI.
    """)