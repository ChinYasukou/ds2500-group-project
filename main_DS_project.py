
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        
        # hypertension proxy
        "hypertension": find_existing_column(df, ["_MICHD"]),
        
        # cholesterol-related
        "cholesterol": find_existing_column(df, ["CHCSCNC1"]),
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
    # bank = not asked or missing
    missing_codes = {
        7: np.nan, 9: np.nan,
        77: np.nan, 88: np.nan, 99: np.nan,
        777: np.nan, 888: np.nan, 999: np.nan,
        7777: np.nan, 8888: np.nan, 9999: np.nan
    }

    clean_df.replace(missing_codes, inplace=True)

    # BMI stored as BMI * 100, normalize it to normal BMI scale
    if "bmi" in clean_df.columns:
        if clean_df["bmi"].dropna().median() > 100:
            clean_df["bmi"] = clean_df["bmi"] / 100

    return clean_df

def main():

    file = "brfss_survey_data_2024.csv"   

    df = load_data(file)

    print("Loaded dataset shape:", df.shape)

    column_map = build_column_map(df)
    print("\nDetected column map:")
    print(column_map)

    clean_df = clean_brfss_data(df, column_map)
    clean_df.to_csv("clean_brfss_data.csv", index=False)

    print("Cleaned dataset saved as clean_brfss_data.csv")

    print("\nCleaned dataset shape:", clean_df.shape)
    print("\nMissing values by column:")
    print(clean_df.isna().sum())


if __name__ == "__main__":
    main()