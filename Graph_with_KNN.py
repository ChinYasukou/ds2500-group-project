import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = "clean_brfss_data.csv"


INCOME_LABELS = {
    1: "<10k",
    2: "10-15k",
    3: "15-20k",
    4: "20-25k",
    5: "25-35k",
    6: "35-50k",
    7: "50-75k",
    8: "75-100k",
    9: "100-150k",
    10: "150-200k",
    11: ">200k"
}

EDUCATION_LABELS = {
    1: "Never attended",
    2: "Elementary",
    3: "Some high school",
    4: "High school graduate",
    5: "Some college",
    6: "College graduate"
}

SEX_LABELS = {
    1: "Male",
    0: "Female"
}

EMPLOYMENT_LABELS = {
    1: "Employed wages",
    2: "Self-employed",
    3: "Out of work >1yr",
    4: "Out of work <1yr",
    5: "Homemaker",
    6: "Student",
    7: "Retired",
    8: "Unable to work"
}

INSURANCE_LABELS = {
    1: "Yes, one plan",
    2: "Yes, multiple",
    3: "No insurance"
}


# =========================
# load data
# =========================
def load_clean_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required_cols = [
        "income",
        "education",
        "employment",
        "insurance",
        "general_health",
        "diabetes",
        "hypertension",
        "cholesterol",
        "age",
        "sex"
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# =========================
# prepare data for plotting
# =========================
def prepare_graph_data(df):
    df = df.copy()

    numeric_cols = [
        "income",
        "education",
        "employment",
        "insurance",
        "general_health",
        "diabetes",
        "hypertension",
        "cholesterol",
        "age",
        "sex"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # keep only valid project values
    df = df[df["income"].isin(INCOME_LABELS.keys())]
    df = df[df["education"].isin(EDUCATION_LABELS.keys())]
    df = df[df["sex"].isin(SEX_LABELS.keys())]
    df = df[df["general_health"].isin([1, 2, 3, 4, 5])]
    df = df[df["diabetes"].isin([0, 1])]
    df = df[df["hypertension"].isin([0, 1])]
    df = df[df["cholesterol"].isin([0, 1])]

    needed = [
        "income",
        "education",
        "general_health",
        "diabetes",
        "hypertension",
        "cholesterol",
        "age",
        "sex"
    ]
    df = df.dropna(subset=needed)

    df["income_label"] = df["income"].map(INCOME_LABELS)
    df["education_label"] = df["education"].map(EDUCATION_LABELS)
    df["sex_label"] = df["sex"].map(SEX_LABELS)
    if "employment" in df.columns:
        df["employment_label"] = df["employment"].map(EMPLOYMENT_LABELS)
    if "insurance" in df.columns:
        df["insurance_label"] = df["insurance"].map(INSURANCE_LABELS)

    return df


# =========================
# helper functions
# =========================
def bar_mean_figure(dataframe, group_col, value_col, title, xlabel, ylabel, order=None):
    grouped = dataframe.groupby(group_col)[value_col].mean()

    if order is not None:
        grouped = grouped.reindex(order)

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


# =========================
# project data visualizations
# =========================
def plot_general_health_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return bar_mean_figure(
        df, "income_label", "general_health",
        "Average General Health by Income",
        "Income Level", "Average General Health Score",
        income_order
    )


def plot_general_health_by_education(df):
    education_order = [EDUCATION_LABELS[k] for k in EDUCATION_LABELS]
    return bar_mean_figure(
        df, "education_label", "general_health",
        "Average General Health by Education",
        "Education Level", "Average General Health Score",
        education_order
    )


def plot_diabetes_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return bar_mean_figure(
        df, "income_label", "diabetes",
        "Diabetes Rate by Income",
        "Income Level", "Diabetes Rate",
        income_order
    )


def plot_diabetes_by_education(df):
    education_order = [EDUCATION_LABELS[k] for k in EDUCATION_LABELS]
    return bar_mean_figure(
        df, "education_label", "diabetes",
        "Diabetes Rate by Education",
        "Education Level", "Diabetes Rate",
        education_order
    )


def plot_hypertension_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return bar_mean_figure(
        df, "income_label", "hypertension",
        "Hypertension Rate by Income",
        "Income Level", "Hypertension Rate",
        income_order
    )


def plot_hypertension_by_education(df):
    education_order = [EDUCATION_LABELS[k] for k in EDUCATION_LABELS]
    return bar_mean_figure(
        df, "education_label", "hypertension",
        "Hypertension Rate by Education",
        "Education Level", "Hypertension Rate",
        education_order
    )


def plot_cholesterol_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return bar_mean_figure(
        df, "income_label", "cholesterol",
        "Cholesterol Rate by Income",
        "Income Level", "Cholesterol Rate",
        income_order
    )


def plot_cholesterol_by_education(df):
    education_order = [EDUCATION_LABELS[k] for k in EDUCATION_LABELS]
    return bar_mean_figure(
        df, "education_label", "cholesterol",
        "Cholesterol Rate by Education",
        "Education Level", "Cholesterol Rate",
        education_order
    )


def plot_diabetes_by_age(df):
    plot_df = df.copy()
    bins = [17, 24, 34, 44, 54, 64, 74, 80]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-80"]
    plot_df["age_group"] = pd.cut(plot_df["age"], bins=bins, labels=labels)
    return bar_mean_figure(
        plot_df, "age_group", "diabetes",
        "Diabetes Rate by Age Group",
        "Age Group", "Diabetes Rate",
        labels
    )


def plot_hypertension_by_age(df):
    plot_df = df.copy()
    bins = [17, 24, 34, 44, 54, 64, 74, 80]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-80"]
    plot_df["age_group"] = pd.cut(plot_df["age"], bins=bins, labels=labels)
    return bar_mean_figure(
        plot_df, "age_group", "hypertension",
        "Hypertension Rate by Age Group",
        "Age Group", "Hypertension Rate",
        labels
    )


# =========================
# summary stats
# =========================
def get_stats(dataframe, column):
    series = dataframe[column].dropna()
    return {
        "column": column,
        "count": int(series.count()),
        "mean": round(series.mean(), 4),
        "median": round(series.median(), 4),
        "mode": series.mode().tolist(),
        "std": round(series.std(), 4)
    }


# =========================
# local test
# =========================
def main():
    df = load_clean_data()
    graph_df = prepare_graph_data(df)

    figures = [
        plot_general_health_by_income(graph_df),
        plot_general_health_by_education(graph_df),
        plot_diabetes_by_income(graph_df),
        plot_diabetes_by_education(graph_df),
        plot_hypertension_by_income(graph_df),
        plot_cholesterol_by_income(graph_df),
        plot_diabetes_by_age(graph_df),
        plot_hypertension_by_age(graph_df),
    ]

    for fig in figures:
        if fig is not None:
            plt.figure(fig.number)
            plt.show()

    print("\nSummary statistics:")
    for col in [
        "income", "education", "general_health",
        "diabetes", "hypertension", "cholesterol", "age"
    ]:
        print(get_stats(graph_df, col))


if __name__ == "__main__":
    main()
