import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from Graph_with_KNN import (
    load_clean_data,
    prepare_graph_data,
    plot_general_health_by_income,
    plot_general_health_by_education,
    plot_diabetes_by_income,
    plot_diabetes_by_education,
    plot_hypertension_by_income,
    plot_hypertension_by_education,
    plot_cholesterol_by_income,
    plot_cholesterol_by_education,
    plot_diabetes_by_age,
    plot_hypertension_by_age,
)


# =========================
# constants (aligned with main_DS_project)
# =========================
NOMINAL_COLS = ["employment", "insurance"]
ORDINAL_COLS = ["income", "education", "age", "sex"]
ALL_FEATURES = ORDINAL_COLS + NOMINAL_COLS

EDUCATION_LABELS_ALT = {
    1: "Never attended", 2: "Grades 1-8",   3: "Grades 9-11",
    4: "Grade 12/GED",   5: "Some college", 6: "College grad",
}
INCOME_LABELS_ALT = {
    1: "<$10k",     2: "$10-15k",   3: "$15-20k",   4: "$20-25k",
    5: "$25-35k",   6: "$35-50k",   7: "$50-75k",   8: "$75-100k",
    9: "$100-150k", 10: "$150-200k", 11: ">$200k",
}
SEX_LABELS_ALT = {1: "Male", 2: "Female"}
EMPLOYMENT_LABELS = {
    1: "Employed wages",   2: "Self-employed",    3: "Out of work >1yr",
    4: "Out of work <1yr", 5: "Homemaker",        6: "Student",
    7: "Retired",          8: "Unable to work",
}
INSURANCE_LABELS = {1: "Yes, one plan", 2: "Yes, multiple", 3: "No insurance"}

TARGET_CLASS_LABELS = {
    "diabetes":     {0: "Not diabetic", 1: "Diabetic"},
    "hypertension": {1: "Has condition", 2: "No condition"},
    "cholesterol":  {1: "Diagnosed",     2: "Not diagnosed"},
}

TARGETS = ["diabetes", "hypertension", "cholesterol"]
K_VALUES = [1, 3, 5, 7, 9, 11]


# =========================
# page setup
# =========================
st.set_page_config(page_title="Health Prediction System", layout="wide")

st.title("Health Prediction System")
st.write("Predict health risks based on socioeconomic factors.")


# =========================
# load data
# =========================
@st.cache_data
def load_project_data():
    return load_clean_data()


@st.cache_data
def load_graph_data():
    d = load_clean_data()
    return prepare_graph_data(d)


df = load_project_data()
graph_df = load_graph_data()


# =========================
# feature engineering (from main_DS_project)
# =========================
def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal columns; keep ordinal/continuous columns numeric."""
    nominal_present = [c for c in NOMINAL_COLS if c in X.columns]
    ordinal_present = [c for c in ORDINAL_COLS if c in X.columns]
    if nominal_present:
        dummies = pd.get_dummies(X[nominal_present].astype(str), prefix=nominal_present)
    else:
        dummies = pd.DataFrame(index=X.index)
    return pd.concat([X[ordinal_present].reset_index(drop=True),
                      dummies.reset_index(drop=True)], axis=1)


def prepare_features_and_target(dataframe, target_col):
    """Select predictor variables and one target variable."""
    available = [c for c in ALL_FEATURES if c in dataframe.columns]
    model_df  = dataframe[available + [target_col]].dropna().copy()
    X_raw     = model_df[available]
    y         = model_df[target_col]
    return encode_features(X_raw), y


def maybe_sample_data(X, y, max_rows=None, random_state=42):
    """Optionally subsample to speed up KNN."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    sampled = X.copy()
    sampled["_target"] = y.values
    sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.drop(columns=["_target"]), sampled["_target"]


# =========================
# scratch KNN (from main_DS_project)
# =========================
def predict_one(X_train, y_train, test_row, k):
    dists = np.sqrt(np.sum((X_train - test_row) ** 2, axis=1))
    nn    = np.argsort(dists)[:k]
    vals, counts = np.unique(y_train[nn], return_counts=True)
    return vals[np.argmax(counts)]


def predict_all(X_train, y_train, X_test, k):
    return np.array([predict_one(X_train, y_train, row, k) for row in X_test])


def evaluate_model(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    )


# =========================
# run full KNN pipeline for one target
# =========================
@st.cache_data
def run_knn_for_target(_df, target_col, k_values, max_rows=10000):
    """Run the full scratch-KNN training pipeline for a single target."""
    X, y = prepare_features_and_target(_df, target_col)
    X, y = maybe_sample_data(X, y, max_rows=max_rows)

    # 60 / 20 / 20 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)

    y_tr = y_train.to_numpy()
    y_va = y_val.to_numpy()
    y_te = y_test.to_numpy()

    # tune k on validation set
    k_results = []
    for k in k_values:
        y_pred = predict_all(X_tr_s, y_tr, X_va_s, k)
        acc, pre, rec, f1 = evaluate_model(y_va, y_pred)
        k_results.append({"k": k, "accuracy": acc, "precision": pre,
                          "recall": rec, "f1": f1})

    k_results_df = pd.DataFrame(k_results)
    best_k = int(k_results_df.loc[k_results_df["f1"].idxmax(), "k"])

    # final evaluation on TEST set with best k
    y_pred_test = predict_all(X_tr_s, y_tr, X_te_s, best_k)
    acc, pre, rec, f1 = evaluate_model(y_te, y_pred_test)

    return {
        "target": target_col,
        "best_k": best_k,
        "accuracy": round(acc, 4),
        "precision": round(pre, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "k_results_df": k_results_df,
        "scaler": scaler,
        "X_tr_s": X_tr_s,
        "y_tr": y_tr,
        "feature_columns": list(X_train.columns),
    }


# =========================
# Altair interactive charts (from main_DS_project)
# =========================
def add_readable_labels(dataframe, target):
    """Add human-readable label columns for Altair charts."""
    dataframe = dataframe.copy()
    dataframe["actual_label"] = dataframe[target].map(
        TARGET_CLASS_LABELS.get(target, {})).fillna(dataframe[target].astype(str))

    if "education" in dataframe.columns:
        dataframe["education_label"] = dataframe["education"].map(EDUCATION_LABELS_ALT)
        dataframe["education_order"] = dataframe["education"]
    if "income" in dataframe.columns:
        dataframe["income_label"] = dataframe["income"].map(INCOME_LABELS_ALT)
        dataframe["income_order"] = dataframe["income"]
    if "sex" in dataframe.columns:
        dataframe["sex_label"] = dataframe["sex"].map(SEX_LABELS_ALT)
    if "age" in dataframe.columns:
        bins   = [17, 24, 34, 44, 54, 64, 74, 80]
        labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-80"]
        dataframe["age_group"] = pd.cut(dataframe["age"], bins=bins, labels=labels).astype(str)
    if "employment" in dataframe.columns:
        dataframe["employment_label"] = dataframe["employment"].map(EMPLOYMENT_LABELS)
    if "insurance" in dataframe.columns:
        dataframe["insurance_label"] = dataframe["insurance"].map(INSURANCE_LABELS)
    return dataframe


def make_actual_chart(dataframe, x_col, x_title, target, sort_col=None):
    """100% stacked bar chart showing actual health outcome proportions."""
    outcome_col = "actual_label"
    counts = (dataframe.groupby([x_col, outcome_col])
                       .size()
                       .reset_index(name="count"))
    totals               = counts.groupby(x_col)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    if sort_col and sort_col in dataframe.columns:
        order_map       = (dataframe[[x_col, sort_col]]
                           .drop_duplicates()
                           .set_index(x_col)[sort_col])
        counts["_sort"] = counts[x_col].map(order_map)
        x_enc = alt.X(f"{x_col}:N", title=x_title,
                      sort=alt.EncodingSortField(field="_sort", order="ascending"))
    else:
        x_enc = alt.X(f"{x_col}:N", title=x_title, sort="ascending")

    class_labels = list(TARGET_CLASS_LABELS.get(target, {}).values())
    color_scale  = alt.Scale(domain=class_labels,
                             range=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    selection    = alt.selection_point(fields=[outcome_col], bind="legend")

    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=x_enc,
            y=alt.Y("proportion:Q", title="Proportion",
                    axis=alt.Axis(format="%")),
            color=alt.Color(f"{outcome_col}:N", title="Actual outcome",
                            scale=color_scale),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[
                alt.Tooltip(f"{x_col}:N",       title=x_title),
                alt.Tooltip(f"{outcome_col}:N", title="Actual outcome"),
                alt.Tooltip("proportion:Q",      title="Proportion", format=".1%"),
                alt.Tooltip("count:Q",           title="Count"),
            ]
        )
        .add_params(selection)
        .properties(width=440, height=280,
                    title=f"Actual {target.capitalize()} by {x_title}")
    )


def build_target_charts(dataframe, target):
    """Build all Altair charts for one health outcome."""
    dataframe = add_readable_labels(dataframe, target)
    charts = []

    if "education_label" in dataframe.columns:
        charts.append(make_actual_chart(
            dataframe, "education_label", "Education Level", target, "education_order"))
    if "income_label" in dataframe.columns:
        charts.append(make_actual_chart(
            dataframe, "income_label", "Household Income", target, "income_order"))
    if "age_group" in dataframe.columns:
        charts.append(make_actual_chart(
            dataframe, "age_group", "Age Group", target))
    if "sex_label" in dataframe.columns:
        charts.append(make_actual_chart(
            dataframe, "sex_label", "Sex", target))
    if "employment_label" in dataframe.columns:
        charts.append(make_actual_chart(
            dataframe, "employment_label", "Employment Status", target))
    if "insurance_label" in dataframe.columns:
        charts.append(make_actual_chart(
            dataframe, "insurance_label", "Insurance Coverage", target))

    rows = []
    for i in range(0, len(charts), 2):
        pair = charts[i:i + 2]
        rows.append(alt.hconcat(*pair).resolve_scale(color="shared"))

    return alt.vconcat(*rows).properties(title=f"── {target.upper()} ──")


# =========================
# run scratch KNN for all targets (cached)
# =========================
@st.cache_data
def run_all_scratch_knn(_df):
    results = {}
    for target in TARGETS:
        if target in _df.columns:
            results[target] = run_knn_for_target(_df, target, K_VALUES, max_rows=10000)
    return results


scratch_results = run_all_scratch_knn(df)


# =========================
# sidebar input
# =========================
st.sidebar.header("Enter Your Information")

income    = st.sidebar.slider("Income Level", 1, 11, 6)
education = st.sidebar.slider("Education Level", 1, 6, 4)
age       = st.sidebar.slider("Age", 18, 80, 30)
sex       = st.sidebar.selectbox("Sex", ["Female", "Male"])
employment = st.sidebar.selectbox(
    "Employment Status",
    list(EMPLOYMENT_LABELS.keys()),
    format_func=lambda x: EMPLOYMENT_LABELS[x],
    index=0
)
insurance = st.sidebar.selectbox(
    "Insurance Coverage",
    list(INSURANCE_LABELS.keys()),
    format_func=lambda x: INSURANCE_LABELS[x],
    index=0
)

sex_value = 1 if sex == "Male" else 0

# build user input with same encoding as training
user_raw = pd.DataFrame([{
    "income": income,
    "education": education,
    "age": age,
    "sex": sex_value,
    "employment": employment,
    "insurance": insurance,
}])
user_encoded = encode_features(user_raw)


# =========================
# prediction helpers
# =========================
def predict_risk_scratch(res, user_enc):
    """Predict using the scratch KNN with the stored training data."""
    # align columns — user_encoded may be missing some one-hot columns
    feature_cols = res["feature_columns"]
    user_aligned = pd.DataFrame(columns=feature_cols)
    user_aligned = pd.concat([user_aligned, user_enc], ignore_index=True).fillna(0)
    # keep only the training columns in the right order
    user_aligned = user_aligned[feature_cols]

    user_scaled = res["scaler"].transform(user_aligned)
    best_k = res["best_k"]

    prediction = predict_one(res["X_tr_s"], res["y_tr"], user_scaled[0], best_k)

    # compute approximate probability using k-neighbor votes
    dists = np.sqrt(np.sum((res["X_tr_s"] - user_scaled[0]) ** 2, axis=1))
    nn = np.argsort(dists)[:best_k]
    neighbor_labels = res["y_tr"][nn]
    risk_score = np.mean(neighbor_labels == 1)

    return int(prediction), float(risk_score)


def risk_label(prediction):
    return "High" if prediction == 1 else "Low"


def risk_color_box(title, prediction, score):
    label = risk_label(prediction)
    percent = round(score * 100, 1)
    if prediction == 1:
        st.error(f"{title}: {label} ({percent}%)")
    else:
        st.success(f"{title}: {label} ({percent}%)")


# =========================
# dataset preview (BMI excluded)
# =========================
st.subheader("Dataset Preview")
preview_cols = [c for c in df.columns if c != "bmi"]
st.dataframe(df[preview_cols].head())


# =========================
# predictions (using scratch KNN — same model as evaluation)
# =========================
pred_diabetes, score_diabetes = predict_risk_scratch(scratch_results["diabetes"], user_encoded)
pred_hypertension, score_hypertension = predict_risk_scratch(scratch_results["hypertension"], user_encoded)
pred_cholesterol, score_cholesterol = predict_risk_scratch(scratch_results["cholesterol"], user_encoded)

st.subheader("Prediction Results")
st.caption("Predictions are based on KNN — finding individuals with similar "
           "socioeconomic profiles and using their outcomes to estimate risk.")

col1, col2, col3 = st.columns(3)

with col1:
    risk_color_box("Diabetes Risk", pred_diabetes, score_diabetes)

with col2:
    risk_color_box("Hypertension Risk", pred_hypertension, score_hypertension)

with col3:
    risk_color_box("Cholesterol Risk", pred_cholesterol, score_cholesterol)


# =========================
# quick interpretation
# =========================
st.subheader("Quick Interpretation")

messages = []
for name, pred in [("diabetes", pred_diabetes),
                    ("hypertension", pred_hypertension),
                    ("cholesterol", pred_cholesterol)]:
    level = "higher" if pred == 1 else "lower"
    messages.append(f"The model predicts a {level} {name} risk for this input profile.")

for message in messages:
    st.write(f"- {message}")


# =========================
# key findings (PPT slides 4 & 6)
# =========================
st.subheader("Key Findings")

st.write(
    "Our analysis reveals a consistent relationship between lower socioeconomic "
    "status and worse health outcomes. These patterns appear across multiple "
    "variables — income, education, employment, and insurance — suggesting they "
    "are not random but reflect structural inequalities in health risk."
)

st.write(
    "Lower income groups tend to show higher rates of diabetes, hypertension, "
    "and cholesterol issues. Similarly, individuals with lower education levels "
    "report worse general health on average."
)


# =========================
# model evaluation (all dynamically computed on test set)
# =========================
st.subheader("Model Evaluation (Scratch KNN)")

st.write("These results come from the from-scratch KNN implementation with "
         "train/validation/test splits and k-value tuning. All metrics are "
         "computed on the held-out test set.")

# summary table — dynamically built from scratch_results
summary_rows = []
for target, res in scratch_results.items():
    summary_rows.append({
        "Target": target.capitalize(),
        "Best k": res["best_k"],
        "Accuracy": res["accuracy"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "F1 Score": res["f1"],
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.write(
    "The model achieves moderate to strong performance across all three outcomes. "
    "Cholesterol and hypertension show higher F1 scores, suggesting these outcomes "
    "are more strongly linked to the socioeconomic features we used. Diabetes is "
    "harder to predict, indicating additional factors beyond socioeconomics play a role."
)

# k-value tuning charts
with st.expander("K-Value Tuning Details"):
    for target, res in scratch_results.items():
        st.write(f"**{target.capitalize()}** — Best k = {res['best_k']}")
        k_df = res["k_results_df"].copy()

        chart = alt.Chart(k_df).mark_line(point=True).encode(
            x=alt.X("k:O", title="k"),
            y=alt.Y("f1:Q", title="F1 Score", scale=alt.Scale(zero=False)),
            tooltip=["k", "accuracy", "precision", "recall", "f1"]
        ).properties(width=500, height=250, title=f"{target.capitalize()} — F1 by k")

        st.altair_chart(chart, use_container_width=True)


# =========================
# model info
# =========================
with st.expander("Model Information"):
    st.write("**Features:** income, education, age, sex (ordinal) + "
             "employment, insurance (one-hot encoded).")
    st.write("**Pipeline:** 60/20/20 train/validation/test split, "
             "StandardScaler normalization, from-scratch KNN with k-value "
             "tuning on validation set, final evaluation on test set.")
    st.write("**Sidebar prediction** uses the same trained model and scaler "
             "as the evaluation — no separate sklearn model.")
    st.write("**Target distribution (full dataset):**")
    for t in TARGETS:
        if t in df.columns:
            dist = df[t].value_counts().to_dict()
            st.write(f"  {t}: {dist}")


# =========================
# visualizations (BMI removed)
# =========================
st.subheader("Data Visualizations")

tab1, tab2, tab3 = st.tabs([
    "Health by Socioeconomic Factors",
    "Disease Rates",
    "Interactive Dashboard",
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_general_health_by_income(graph_df))
        st.pyplot(plot_diabetes_by_age(graph_df))
    with col2:
        st.pyplot(plot_general_health_by_education(graph_df))
        st.pyplot(plot_hypertension_by_age(graph_df))

with tab2:
    st.caption("Lower income groups consistently show higher rates of chronic disease.")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_diabetes_by_income(graph_df))
        st.pyplot(plot_hypertension_by_income(graph_df))
        st.pyplot(plot_cholesterol_by_income(graph_df))
    with col2:
        st.pyplot(plot_diabetes_by_education(graph_df))
        st.pyplot(plot_hypertension_by_education(graph_df))
        st.pyplot(plot_cholesterol_by_education(graph_df))

with tab3:
    st.write("Interactive Altair charts showing actual health outcome distributions "
             "across socioeconomic groups. Hover for details, click the legend to filter.")

    alt.data_transformers.enable("default", max_rows=None)

    selected_target = st.selectbox("Select health outcome:", TARGETS,
                                   format_func=lambda x: x.capitalize())

    if selected_target in df.columns:
        altair_chart = build_target_charts(df, selected_target)
        st.altair_chart(altair_chart, use_container_width=True)


# =========================
# conclusions & limitations (PPT slide 6)
# =========================
st.subheader("Conclusions & Limitations")

st.write(
    "Clear and consistent relationships exist between socioeconomic status and "
    "health outcomes. However, these patterns alone are not strong enough for "
    "highly accurate prediction."
)

st.write(
    "Health outcomes are also influenced by behavioral factors, access to care, "
    "and biological differences — none of which are captured in this dataset. "
    "Socioeconomic data can help identify at-risk populations, but effective "
    "prediction and policy require more comprehensive health information."
)


# =========================
# footer
# =========================
st.write("---")
st.write("DS2500 Final Project — Lucy Pesek, Sharon Wu, Zineb Laghzaoui, Jinghao Shen (Frank)")
