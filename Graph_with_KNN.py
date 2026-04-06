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


# =========================
# load data
# =========================
def load_clean_data():
    df = pd.read_csv(DATA_FILE)

    required_cols = [
        "income",
        "education",
        "employment",
        "insurance",
        "bmi",
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
        "bmi",
        "general_health",
        "diabetes",
        "hypertension",
        "cholesterol",
        "age",
        "sex"
    ]

    for col in numeric_cols:
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
        "bmi",
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

    return df


# =========================
# helper functions
# =========================
def boxplot_figure(dataframe, group_col, value_col, title, xlabel, ylabel, order=None):
    plot_df = dataframe.copy()

    if order is not None:
        plot_df[group_col] = pd.Categorical(
            plot_df[group_col],
            categories=order,
            ordered=True
        )
        plot_df = plot_df.sort_values(group_col)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.boxplot(column=value_col, by=group_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("")
    fig.tight_layout()
    return fig


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


def scatter_figure(dataframe, x_col, y_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dataframe[x_col], dataframe[y_col], alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


# =========================
# project data visualizations
# =========================
def plot_bmi_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return boxplot_figure(
        df,
        "income_label",
        "bmi",
        "BMI by Income Level",
        "Income Level",
        "BMI",
        income_order
    )


def plot_bmi_by_education(df):
    education_order = [EDUCATION_LABELS[k] for k in EDUCATION_LABELS]
    return boxplot_figure(
        df,
        "education_label",
        "bmi",
        "BMI by Education Level",
        "Education Level",
        "BMI",
        education_order
    )


def plot_general_health_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return bar_mean_figure(
        df,
        "income_label",
        "general_health",
        "Average General Health by Income",
        "Income Level",
        "Average General Health Score",
        income_order
    )


def plot_general_health_by_education(df):
    education_order = [EDUCATION_LABELS[k] for k in EDUCATION_LABELS]
    return bar_mean_figure(
        df,
        "education_label",
        "general_health",
        "Average General Health by Education",
        "Education Level",
        "Average General Health Score",
        education_order
    )


def plot_diabetes_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return bar_mean_figure(
        df,
        "income_label",
        "diabetes",
        "Diabetes Rate by Income",
        "Income Level",
        "Diabetes Rate",
        income_order
    )


def plot_diabetes_by_education(df):
    education_order = [EDUCATION_LABELS[k] for k in EDUCATION_LABELS]
    return bar_mean_figure(
        df,
        "education_label",
        "diabetes",
        "Diabetes Rate by Education",
        "Education Level",
        "Diabetes Rate",
        education_order
    )


def plot_hypertension_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return bar_mean_figure(
        df,
        "income_label",
        "hypertension",
        "Hypertension Rate by Income",
        "Income Level",
        "Hypertension Rate",
        income_order
    )


def plot_cholesterol_by_income(df):
    income_order = [INCOME_LABELS[k] for k in INCOME_LABELS]
    return bar_mean_figure(
        df,
        "income_label",
        "cholesterol",
        "Cholesterol Rate by Income",
        "Income Level",
        "Cholesterol Rate",
        income_order
    )


def plot_age_vs_bmi(df):
    return scatter_figure(
        df,
        "age",
        "bmi",
        "Age vs BMI",
        "Age",
        "BMI"
    )


def plot_bmi_by_sex(df):
    sex_order = ["Male", "Female"]
    return boxplot_figure(
        df,
        "sex_label",
        "bmi",
        "BMI by Sex",
        "Sex",
        "BMI",
        sex_order
    )


# =========================
# KNN evaluation figures
# =========================
def plot_accuracy_comparison(diabetes_acc=0.887, hypertension_acc=0.925, cholesterol_acc=0.956):
    models = ["Diabetes", "Hypertension", "Cholesterol"]
    accuracy = [diabetes_acc, hypertension_acc, cholesterol_acc]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, accuracy)
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(prediction_file="knn_diabetes_predictions_full.csv"):
    df = pd.read_csv(prediction_file)

    actual = df["actual_diabetes"].value_counts().sort_index()
    predicted = df["predicted_diabetes"].value_counts().sort_index()

    actual_no = actual.get(0, 0)
    actual_yes = actual.get(1, 0)
    pred_no = predicted.get(0, 0)
    pred_yes = predicted.get(1, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([0, 1], [actual_no, actual_yes], width=0.4, label="Actual")
    ax.bar([0.4, 1.4], [pred_no, pred_yes], width=0.4, label="Predicted")
    ax.set_xticks([0.2, 1.2])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_title("Actual vs Predicted (Diabetes)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_count(prediction_file="knn_diabetes_predictions_full.csv"):
    df = pd.read_csv(prediction_file)

    errors = (df["actual_diabetes"] != df["predicted_diabetes"]).astype(int)
    counts = errors.value_counts()

    correct_count = counts.get(0, 0)
    wrong_count = counts.get(1, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Correct", "Wrong"], [correct_count, wrong_count])
    ax.set_title("Prediction Errors")
    fig.tight_layout()
    return fig


def plot_cumulative_correct(prediction_file="knn_diabetes_predictions_full.csv"):
    df = pd.read_csv(prediction_file)

    correct = (df["actual_diabetes"] == df["predicted_diabetes"]).astype(int)
    cumulative = correct.cumsum()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative)
    ax.set_title("Cumulative Correct Predictions")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Correct Predictions")
    fig.tight_layout()
    return fig


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
        plot_bmi_by_income(graph_df),
        plot_bmi_by_education(graph_df),
        plot_general_health_by_income(graph_df),
        plot_general_health_by_education(graph_df),
        plot_diabetes_by_income(graph_df),
        plot_diabetes_by_education(graph_df),
        plot_hypertension_by_income(graph_df),
        plot_cholesterol_by_income(graph_df),
        plot_age_vs_bmi(graph_df),
        plot_bmi_by_sex(graph_df)
    ]

    for fig in figures:
        plt.figure(fig.number)
        plt.show()

    print("\nSummary statistics:")
    for col in [
        "bmi",
        "income",
        "education",
        "general_health",
        "diabetes",
        "hypertension",
        "cholesterol",
        "age"
    ]:
        print(get_stats(graph_df, col))


if __name__ == "__main__":
    main()