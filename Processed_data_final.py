import pandas as pd
import numpy as np


# columns used in the final project
SELECTED_COLUMNS = [
    "_BMI5",
    "INCOME3",
    "EDUCA",
    "EMPLOY1",
    "PRIMINS2",
    "GENHLTH",
    "DIABETE4",
    "_MICHD",
    "CHCSCNC1",
    "_AGE80",
    "SEXVAR"
]


def load_raw_data():
    """
    Load only the columns needed for the project.
    """
    return pd.read_csv("brfss_survey_data_2024.csv", usecols=SELECTED_COLUMNS)


def replace_missing_codes(df):
    """
    Replace survey missing-value codes with NaN.
    """
    df = df.copy()

    missing_codes = [
        7, 8, 9,
        77, 88, 99,
        777, 888, 999,
        7777, 8888, 9999
    ]

    df = df.replace(missing_codes, np.nan)
    return df


def recode_variables(df):
    """
    Recode variables into cleaner formats for analysis and modeling.
    """
    df = df.copy()

    # BMI is stored as an integer scaled by 100
    df["_BMI5"] = df["_BMI5"] / 100

    # keep only realistic BMI values
    df.loc[(df["_BMI5"] < 10) | (df["_BMI5"] > 80), "_BMI5"] = np.nan

    # sex: 1 = male, 2 = female -> convert to binary
    df["SEXVAR"] = df["SEXVAR"].map({
        1: 1,   # male
        2: 0    # female
    })

    # hypertension proxy: 1 = yes, 2 = no
    df["_MICHD"] = df["_MICHD"].map({
        1: 1,
        2: 0
    })

    # cholesterol indicator: 1 = yes, 2 = no
    df["CHCSCNC1"] = df["CHCSCNC1"].map({
        1: 1,
        2: 0
    })

    # diabetes:
    # 1 = yes
    # 2 = yes, but only during pregnancy
    # 3 = no
    # 4 = prediabetes / borderline
    #
    # for a simple binary model:
    # treat clear yes values as 1
    # treat no / borderline as 0
    df["DIABETE4"] = df["DIABETE4"].map({
        1: 1,
        2: 1,
        3: 0,
        4: 0
    })

    return df


def fill_missing_values(df):
    """
    Fill missing values with mode so category values stay valid.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def rename_columns(df):
    """
    Rename columns into final project-friendly names.
    """
    return df.rename(columns={
        "_BMI5": "bmi",
        "INCOME3": "income",
        "EDUCA": "education",
        "EMPLOY1": "employment",
        "PRIMINS2": "insurance",
        "GENHLTH": "general_health",
        "DIABETE4": "diabetes",
        "_MICHD": "hypertension",
        "CHCSCNC1": "cholesterol",
        "_AGE80": "age",
        "SEXVAR": "sex"
    })


def clean_data():
    """
    Full cleaning pipeline.
    """
    df = load_raw_data()
    df = replace_missing_codes(df)
    df = recode_variables(df)
    df = fill_missing_values(df)
    df = rename_columns(df)
    return df


def print_quality_check(df):
    """
    Print quick checks so we can verify the cleaned data.
    """
    print("Cleaned dataset created successfully.")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values by column:")
    print(df.isna().sum())

    print("\nValue counts check:")

    print("\nDiabetes:")
    print(df["diabetes"].value_counts(dropna=False).sort_index())

    print("\nHypertension:")
    print(df["hypertension"].value_counts(dropna=False).sort_index())

    print("\nCholesterol:")
    print(df["cholesterol"].value_counts(dropna=False).sort_index())

    print("\nSex:")
    print(df["sex"].value_counts(dropna=False).sort_index())

    print("\nBMI summary:")
    print(df["bmi"].describe())


def main():
    df = clean_data()
    df.to_csv("clean_brfss_data.csv", index=False)
    print_quality_check(df)


if __name__ == "__main__":
    main()