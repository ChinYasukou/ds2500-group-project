import pandas as pd

# Columns to keep (aligned with your project)
cols = [
    "_BMI5",
    "INCOME3",
    "EDUCA",
    "EMPLOY1",
    "PRIMINS2",
    "GENHLTH",
    "DIABETE4",
    "_MICHD",        # hypertension proxy
    "CHCSCNC1",      # cholesterol-related
    "_AGE80",
    "SEXVAR"
]

# Load dataset
df = pd.read_csv("brfss_survey_data_2024.csv", usecols=cols)

# Replace missing codes with NA
df = df.replace([7, 9, 77, 99], pd.NA)

# Fill binary columns with mode FIRST
binary_cols = ["PRIMINS2", "DIABETE4", "SEXVAR", "_MICHD", "CHCSCNC1"]
for col in binary_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill rest numeric columns with averages
df = df.fillna(df.mean(numeric_only=True))

# Fix BMI scale
df["_BMI5"] = df["_BMI5"] / 100

# Convert binary variables to 0/1 for better classification
df["PRIMINS2"] = df["PRIMINS2"].replace({1: 1, 2: 0})
df["DIABETE4"] = df["DIABETE4"].replace({1: 1, 2: 0})
df["SEXVAR"] = df["SEXVAR"].replace({1: 1, 2: 0})
df["_MICHD"] = df["_MICHD"].replace({1: 1, 2: 0})
df["CHCSCNC1"] = df["CHCSCNC1"].replace({1: 1, 2: 0})

# Save cleaned dataset
df.to_csv("cleaned_brfss.csv", index=False)

print("Preprocessing complete, cleaned_brfss.csv created")
