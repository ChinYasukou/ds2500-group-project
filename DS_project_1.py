#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. Loads BRFSS survey data from a CSV file
2. Selects relevant socioeconomic and health columns
3. Cleans missing/special coded values
4. Creates health outcome categories
"""

import pandas as pd
import numpy as np

def load_data(file):
    return pd.read_csv(file, sep=",", skiprows=1, low_memory=False)

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
        "income": find_existing_column(df, ["INCOME3"]),
        "education": find_existing_column(df, ["EDUCA"]),
        "employment": find_existing_column(df, ["EMPLOY1"]),
        "insurance": find_existing_column(df, ["PERSDOC3"]),
        "bmi": find_existing_column(df, ["_BMI5"]),
        "diabetes": find_existing_column(df, ["DIABETE4"]),
        "hypertension": find_existing_column(df, ["BPHIGH6", "BPHIGH4", "HYPERTEN"]),
        "cholesterol": find_existing_column(df, ["TOLDHI3", "TOLDHI2", "CHOLHIGH"]),
        "age": find_existing_column(df, ["_AGE80"]),
        "sex": find_existing_column(df, ["SEXVAR"])
    }
    return column_map


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


def main():

    file = "brfss_survey_data_2024.csv"   

    df = load_data(file)

    print("Loaded dataset shape:", df.shape)
    print("\nFirst 20 columns in dataset:")
    print(df.columns[:20].tolist())

    column_map = build_column_map(df)
    print("\nDetected column map:")
    print(column_map)

    clean_df = clean_brfss_data(df, column_map)
    clean_df = create_outcome_variables(clean_df)
    clean_df.to_csv("clean_brfss_data.csv", index=False)

    print("Cleaned dataset saved as clean_brfss_data.csv")

    print("\nCleaned dataset shape:", clean_df.shape)
    print("\nMissing values by column:")
    print(clean_df.isna().sum())


if __name__ == "__main__":
    main()