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

# Convert binary variables to 0/1
df["PRIMINS2"] = df["PRIMINS2"].replace({1: 1, 2: 0})
df["DIABETE4"] = df["DIABETE4"].replace({
    1: 1,   # yes
    2: 0,   # no
    3: 0,   # prediabetes → treat as no
    4: 0    # borderline → treat as no
})
df["SEXVAR"] = df["SEXVAR"].replace({1: 1, 2: 0})
df["_MICHD"] = df["_MICHD"].replace({1: 1, 2: 0})
df["CHCSCNC1"] = df["CHCSCNC1"].replace({1: 1, 2: 0})

# Fill binary columns with mode FIRST
binary_cols = ["PRIMINS2", "DIABETE4", "SEXVAR", "_MICHD", "CHCSCNC1"]
for col in binary_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill remaining numeric columns with mean
df = df.fillna(df.mean(numeric_only=True))

# Fix BMI scale
df["_BMI5"] = df["_BMI5"] / 100


df.rename(columns={
    "INCOME3": "income",
    "EDUCA": "education",
    "EMPLOY1": "employment",
    "PRIMINS2": "insurance",
    "DIABETE4": "diabetes",
    "_MICHD": "hypertension",
    "CHCSCNC1": "cholesterol",
    "_AGE80": "age",
    "SEXVAR": "sex"
}, inplace=True)

# Save cleaned dataset
df.to_csv("clean_brfss_data.csv", index=False)

print("Preprocessing complete — cleaned_brfss.csv created")
