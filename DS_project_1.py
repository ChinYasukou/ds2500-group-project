#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Team project: Socioeconomic factors and health outcomes in BRFSS

This script:
1. Loads BRFSS survey data from a CSV file
2. Selects relevant socioeconomic and health columns
3. Cleans missing/special coded values
4. Creates health outcome categories
5. Produces descriptive statistics and visualizations
6. Builds classification models to predict health outcomes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------

def load_data(file_path):
    """
    Load BRFSS data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        Loaded dataset.
    """
    return pd.read_csv(file_path, low_memory=False)


# ---------------------------------------------------
# 2. COLUMN HELPERS
# ---------------------------------------------------

def find_existing_column(df, possible_names):
    """
    Return the first column name from possible_names that exists in df.

    Parameters
    ----------
    df : DataFrame
        Input dataset.
    possible_names : list
        Possible column names to search for.

    Returns
    -------
    str or None
        Matching column name if found, otherwise None.
    """
    for col in possible_names:
        if col in df.columns:
            return col
    return None


def build_column_map(df):
    """
    Build a column map using likely BRFSS variable names.

    Adjust this list if your dataset uses different column names.

    Parameters
    ----------
    df : DataFrame
        Input dataset.

    Returns
    -------
    dict
        Dictionary mapping friendly names to actual dataset column names.
    """
    column_map = {
        "income": find_existing_column(df, ["INCOME3", "INCOME2", "_INCOMG1"]),
        "education": find_existing_column(df, ["EDUCA", "_EDUCAG"]),
        "employment": find_existing_column(df, ["EMPLOY1", "EMPLOY"]),
        "insurance": find_existing_column(df, ["HLTHPLN1", "PERSDOC3"]),
        "bmi": find_existing_column(df, ["_BMI5", "BMI5", "BMI"]),
        "diabetes": find_existing_column(df, ["DIABETE4", "DIABETE3", "DIABETES"]),
        "hypertension": find_existing_column(df, ["BPHIGH6", "BPHIGH4", "HYPERTEN"]),
        "cholesterol": find_existing_column(df, ["TOLDHI3", "TOLDHI2", "CHOLHIGH"]),
        "age": find_existing_column(df, ["_AGE80", "_AGEG5YR", "AGE"]),
        "sex": find_existing_column(df, ["SEXVAR", "SEX"])
    }
    return column_map


# ---------------------------------------------------
# 3. CLEAN DATA
# ---------------------------------------------------

def clean_brfss_data(df, column_map):
    """
    Select relevant columns and clean common BRFSS coded missing values.

    Parameters
    ----------
    df : DataFrame
        Raw BRFSS dataset.
    column_map : dict
        Mapping of friendly names to real dataset column names.

    Returns
    -------
    DataFrame
        Cleaned dataframe with renamed columns.
    """
    selected = {k: v for k, v in column_map.items() if v is not None}
    clean_df = df[list(selected.values())].copy()
    clean_df.rename(columns={v: k for k, v in selected.items()}, inplace=True)

    # Common BRFSS nonresponse / unknown / refused codes
    missing_codes = {
        7: np.nan, 9: np.nan,
        77: np.nan, 88: np.nan, 99: np.nan,
        777: np.nan, 888: np.nan, 999: np.nan,
        7777: np.nan, 8888: np.nan, 9999: np.nan
    }

    clean_df.replace(missing_codes, inplace=True)

    # Convert to numeric where possible
    for col in clean_df.columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="ignore")

    # BMI in BRFSS is often stored as BMI * 100
    if "bmi" in clean_df.columns:
        if clean_df["bmi"].dropna().median() > 100:
            clean_df["bmi"] = clean_df["bmi"] / 100

    return clean_df


# ---------------------------------------------------
# 4. CREATE TARGET VARIABLES
# ---------------------------------------------------

def create_outcome_variables(df):
    """
    Create binary health outcome columns for modeling.

    Parameters
    ----------
    df : DataFrame
        Cleaned dataset.

    Returns
    -------
    DataFrame
        Dataframe with added target variables.
    """
    df = df.copy()

    if "bmi" in df.columns:
        df["obesity_risk"] = np.where(df["bmi"] >= 30, 1, 0)
        df.loc[df["bmi"].isna(), "obesity_risk"] = np.nan

    # BRFSS diabetes coding often:
    # 1 = yes, 2 = yes but female told only during pregnancy, 3 = no, etc.
    if "diabetes" in df.columns:
        df["diabetes_binary"] = np.where(df["diabetes"] == 1, 1,
                                  np.where(df["diabetes"] == 3, 0, np.nan))

    # BRFSS hypertension coding often:
    # 1 = yes, 3 = no
    if "hypertension" in df.columns:
        df["hypertension_binary"] = np.where(df["hypertension"] == 1, 1,
                                      np.where(df["hypertension"] == 3, 0, np.nan))

    # BRFSS high cholesterol coding often:
    # 1 = yes, 2 = no
    if "cholesterol" in df.columns:
        df["cholesterol_binary"] = np.where(df["cholesterol"] == 1, 1,
                                     np.where(df["cholesterol"] == 2, 0, np.nan))

    return df


# ---------------------------------------------------
# 5. DESCRIPTIVE STATS
# ---------------------------------------------------

def descriptive_stats(df):
    """
    Print descriptive statistics for the main variables.

    Parameters
    ----------
    df : DataFrame
        Cleaned dataset.
    """
    print("\nDESCRIPTIVE STATISTICS")
    print(df.describe(include="all"))


# ---------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------

def plot_income_vs_bmi(df):
    """
    Create a boxplot of BMI by income category.

    Parameters
    ----------
    df : DataFrame
        Dataset containing income and bmi.
    """
    if "income" in df.columns and "bmi" in df.columns:
        temp = df[["income", "bmi"]].dropna()
        plt.figure(figsize=(10, 6))
        temp.boxplot(column="bmi", by="income")
        plt.title("BMI by Income Category")
        plt.suptitle("")
        plt.xlabel("Income Category")
        plt.ylabel("BMI")
        plt.tight_layout()
        plt.show()


def plot_employment_vs_obesity(df):
    """
    Create a bar chart of obesity rate by employment category.

    Parameters
    ----------
    df : DataFrame
        Dataset containing employment and obesity_risk.
    """
    if "employment" in df.columns and "obesity_risk" in df.columns:
        temp = df[["employment", "obesity_risk"]].dropna()
        rates = temp.groupby("employment")["obesity_risk"].mean().sort_index()

        plt.figure(figsize=(10, 6))
        rates.plot(kind="bar")
        plt.title("Obesity Rate by Employment Status")
        plt.xlabel("Employment Category")
        plt.ylabel("Proportion Obese")
        plt.tight_layout()
        plt.show()


def plot_education_vs_diabetes(df):
    """
    Create a bar chart of diabetes rate by education category.

    Parameters
    ----------
    df : DataFrame
        Dataset containing education and diabetes_binary.
    """
    if "education" in df.columns and "diabetes_binary" in df.columns:
        temp = df[["education", "diabetes_binary"]].dropna()
        rates = temp.groupby("education")["diabetes_binary"].mean().sort_index()

        plt.figure(figsize=(10, 6))
        rates.plot(kind="bar")
        plt.title("Diabetes Rate by Education Level")
        plt.xlabel("Education Category")
        plt.ylabel("Proportion with Diabetes")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------
# 7. MODELING
# ---------------------------------------------------

def run_classification_models(df, target_column):
    """
    Train logistic regression and random forest models for one target.

    Parameters
    ----------
    df : DataFrame
        Dataset containing predictors and target.
    target_column : str
        Name of target column.

    Returns
    -------
    None
    """
    feature_cols = ["income", "education", "employment", "insurance", "age", "sex"]
    feature_cols = [col for col in feature_cols if col in df.columns]

    model_df = df[feature_cols + [target_column]].dropna()

    if model_df.empty:
        print(f"\nSkipping {target_column}: no usable rows after dropping missing values.")
        return

    X = model_df[feature_cols]
    y = model_df[target_column].astype(int)

    categorical_cols = [col for col in feature_cols if col in ["income", "education", "employment", "insurance", "sex"]]
    numeric_cols = [col for col in feature_cols if col in ["age"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols)
        ],
        remainder="drop"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n{'=' * 60}")
    print(f"TARGET: {target_column}")
    print(f"{'=' * 60}")

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{model_name}")
        print(f"Accuracy: {acc:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(classification_report(y_test, y_pred))


# ---------------------------------------------------
# 8. MAIN
# ---------------------------------------------------

def main():
    """
    Run the full BRFSS health disparity analysis pipeline.
    """
    file_path = "brfss_survey_data_2024"   # change this to your exported CSV filename

    df = load_data(file_path)

    print("Loaded dataset shape:", df.shape)
    print("\nFirst 20 columns in dataset:")
    print(df.columns[:20].tolist())

    column_map = build_column_map(df)
    print("\nDetected column map:")
    print(column_map)

    clean_df = clean_brfss_data(df, column_map)
    clean_df = create_outcome_variables(clean_df)

    print("\nCleaned dataset shape:", clean_df.shape)
    print("\nMissing values by column:")
    print(clean_df.isna().sum())

    descriptive_stats(clean_df)

    # Plots
    plot_income_vs_bmi(clean_df)
    plot_employment_vs_obesity(clean_df)
    plot_education_vs_diabetes(clean_df)

    # Models
    for target in ["obesity_risk", "diabetes_binary", "hypertension_binary", "cholesterol_binary"]:
        if target in clean_df.columns:
            run_classification_models(clean_df, target)


if __name__ == "__main__":
    main()