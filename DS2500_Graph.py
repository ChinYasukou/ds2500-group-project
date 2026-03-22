import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("brfss_survey_data_2024.csv")

# =========================
# 2. LIGHT PREPROCESSING
# =========================

# Keep only the columns needed for this milestone
cols = ["_BMI5", "INCOME3", "EDUCA", "GENHLTH", "DIABETE4"]
df = df[cols].copy()

# Drop rows with missing values in the selected columns
df = df.dropna()

# Convert BMI to the correct scale
df["_BMI5"] = df["_BMI5"] / 100

# Keep only valid education categories used in the graphs
df = df[df["EDUCA"].isin([1, 2, 3, 4, 5, 6, 9])]

# Keep only valid income categories used in the graphs
df = df[df["INCOME3"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 77, 99])]

# Keep only valid general health responses
df = df[df["GENHLTH"].isin([1, 2, 3, 4, 5])]

# =========================
# 3. CHECK DIABETES CODING
# =========================

print("Unique DIABETE4 values before recoding:")
print(df["DIABETE4"].value_counts(dropna=False).sort_index())

# This version assumes:
# 1 = yes
# 3 = no
df_diabetes = df[df["DIABETE4"].isin([1, 3])].copy()
df_diabetes["DIABETE4"] = df_diabetes["DIABETE4"].replace({1: 1, 3: 0})

print("\nUnique DIABETE4 values after recoding:")
print(df_diabetes["DIABETE4"].value_counts(dropna=False).sort_index())

# =========================
# 4. STATISTICS FUNCTION
# =========================

def get_stats(dataframe, column):
    series = dataframe[column].dropna()
    return {
        "column": column,
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().tolist(),
        "std": series.std()
    }

# =========================
# 5. PLOTTING FUNCTIONS
# =========================

def make_boxplot(dataframe, group_col, value_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    dataframe.boxplot(column=value_col, by=group_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle("")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def make_grouped_bar(dataframe, group_col, value_col, title, xlabel, ylabel):
    grouped = dataframe.groupby(group_col)[value_col].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def make_scatter(dataframe, x_col, y_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dataframe[x_col], dataframe[y_col], alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# =========================
# 6. RUN ANALYSIS
# =========================

make_boxplot(
    df,
    "INCOME3",
    "_BMI5",
    "BMI by Income Level",
    "Income Level",
    "BMI"
)

make_boxplot(
    df,
    "EDUCA",
    "_BMI5",
    "BMI by Education Level",
    "Education Level",
    "BMI"
)

make_grouped_bar(
    df,
    "INCOME3",
    "GENHLTH",
    "General Health by Income",
    "Income Level",
    "Average General Health Score"
)

make_grouped_bar(
    df,
    "EDUCA",
    "GENHLTH",
    "General Health by Education",
    "Education Level",
    "Average General Health Score"
)

make_grouped_bar(
    df_diabetes,
    "INCOME3",
    "DIABETE4",
    "Diabetes Rate by Income",
    "Income Level",
    "Diabetes Rate"
)

make_grouped_bar(
    df_diabetes,
    "EDUCA",
    "DIABETE4",
    "Diabetes Rate by Education",
    "Education Level",
    "Diabetes Rate"
)

# =========================
# 7. PRINT SUMMARY STATISTICS
# =========================

print("\nSummary statistics for selected variables:")
print(get_stats(df, "_BMI5"))
print(get_stats(df, "INCOME3"))
print(get_stats(df, "EDUCA"))
print(get_stats(df, "GENHLTH"))
print(get_stats(df_diabetes, "DIABETE4"))