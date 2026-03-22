import pandas as pd

# Download dataset from kaggle as csv file
df = pd.read_csv("brfss_survey_data_2024.csv")

# relevant columns to our project
cols = [
    "_BMI5",
    "INCOME3",
    "EDUCA",
    "EMPLOY1",
    "PRIMINS2",
    "GENHLTH",
    "DIABETE4"
]
df = df[cols]

# Handle missing values
df = df.replace([7, 9, 77, 99], pd.NA)
df = df.dropna()

# adjusted BMI scale
df["_BMI5"] = df["_BMI5"] / 100

# categorical variables organization
df["PRIMINS2"] = df["PRIMINS2"].map({1: 1, 2: 0})
df["DIABETE4"] = df["DIABETE4"].replace({1: 1, 2: 0})

# Save cleaned dataset
df.to_csv("cleaned_brfss.csv", index=False)

print("Preprocessing complete. File saved as cleaned_brfss.csv")
