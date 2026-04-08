#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data(file):
    """Load raw BRFSS CSV file."""
    return pd.read_csv(file, sep=",", skiprows=1, low_memory=False)


def load_clean_data(file):
    """Load already-cleaned CSV file."""
    return pd.read_csv(file, low_memory=False)


def find_existing_column(df, possible_names):
    """Return the first column name from possible_names that exists in df."""
    for col in possible_names:
        if col in df.columns:
            return col
    return None


def build_column_map(df):
    """Build a column map using likely BRFSS variable names."""
    return {
        "income":       find_existing_column(df, ["INCOME3"]),
        "education":    find_existing_column(df, ["EDUCA"]),
        "employment":   find_existing_column(df, ["EMPLOY1"]),
        "insurance":    find_existing_column(df, ["PERSDOC3"]),
        "diabetes":     find_existing_column(df, ["DIABETE4"]),
        "hypertension": find_existing_column(df, ["_MICHD"]),
        "cholesterol":  find_existing_column(df, ["CHCSCNC1"]),
        "age":          find_existing_column(df, ["_AGE80"]),
        "sex":          find_existing_column(df, ["SEXVAR"]),
    }


def clean_brfss_data(df, column_map):
    """
    Select relevant columns, replace BRFSS missing-value codes with NaN,
    then DROP rows that have any missing values in the columns we care about.
    """
    selected = {k: v for k, v in column_map.items() if v is not None}
    clean_df = df[list(selected.values())].copy()
    clean_df.rename(columns={v: k for k, v in selected.items()}, inplace=True)

    for col in clean_df.columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Multi-digit sentinel codes → NaN (safe to apply globally)
    global_missing = {
        77: np.nan,   88: np.nan,  99: np.nan,
        777: np.nan,  888: np.nan, 999: np.nan,
        7777: np.nan, 8888: np.nan, 9999: np.nan,
    }
    clean_df.replace(global_missing, inplace=True)

    # Single-digit sentinels: only on columns where 7/9 are NOT valid category codes
    single_digit_cols = ["income", "education", "diabetes",
                         "hypertension", "cholesterol", "sex", "insurance"]
    for col in single_digit_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].replace({7: np.nan, 9: np.nan})

    # Employment: 7 = Retired (valid), 9 = Refused (sentinel)
    if "employment" in clean_df.columns:
        clean_df["employment"] = clean_df["employment"].replace({9: np.nan})

    before = len(clean_df)
    clean_df.dropna(inplace=True)
    after = len(clean_df)
    print(f"Dropped {before - after} rows with missing values ({after} rows remain).")

    # Recode diabetes to binary: 1=Yes → 1, 3=No → 0, drop gestational/pre-diabetes
    if "diabetes" in clean_df.columns:
        before_diab = len(clean_df)
        clean_df = clean_df[clean_df["diabetes"].isin([1, 3])].copy()
        clean_df["diabetes"] = clean_df["diabetes"].map({1: 1, 3: 0})
        print(f"Recoded diabetes to binary. "
              f"Dropped {before_diab - len(clean_df)} rows (gestational/pre-diabetes).")

    return clean_df


def save_clean_model_file(df, file_name="clean_brfss_data.csv"):
    """Save cleaned dataframe to CSV."""
    df.to_csv(file_name, index=False)
    print(f"Saved cleaned data as {file_name}")


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

NOMINAL_COLS = ["employment", "insurance"]
ORDINAL_COLS = ["income", "education", "age", "sex"]


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


def prepare_features_and_target(df, target_col):
    """Select predictor variables and one target variable."""
    feature_cols = ORDINAL_COLS + NOMINAL_COLS
    available    = [c for c in feature_cols if c in df.columns]
    model_df     = df[available + [target_col]].dropna().copy()
    X_raw        = model_df[available]
    y            = model_df[target_col]
    return encode_features(X_raw), y


def split_train_validation_test(X, y, train_size=0.6, val_size=0.2,
                                test_size=0.2, random_state=42):
    """Split into train / validation / test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_size), random_state=random_state, stratify=y)
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_datasets(X_train, X_val, X_test):
    """Standardize all features using the training set only."""
    scaler = StandardScaler()
    return (scaler.fit_transform(X_train),
            scaler.transform(X_val),
            scaler.transform(X_test))


def maybe_sample_data(X, y, max_rows=None, random_state=42):
    """Optionally subsample to speed up KNN."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    sampled = X.copy()
    sampled["_target"] = y.values
    sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.drop(columns=["_target"]), sampled["_target"]


# ─────────────────────────────────────────────
# KNN (from scratch)
# ─────────────────────────────────────────────

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


def test_k_values(X_train, y_train, X_val, y_val, k_values):
    results = []
    for k in k_values:
        y_pred = predict_all(X_train, y_train, X_val, k)
        acc, pre, rec, f1 = evaluate_model(y_val, y_pred)
        results.append({"k": k, "accuracy": acc, "precision_weighted": pre,
                        "recall_weighted": rec, "f1_weighted": f1})
        print(f"  k={k} | Acc={acc:.4f} | Pre={pre:.4f} | Rec={rec:.4f} | F1={f1:.4f}")
    df_r = pd.DataFrame(results)
    return df_r, int(df_r.loc[df_r["f1_weighted"].idxmax(), "k"])


# ─────────────────────────────────────────────
# SAVING RESULTS
# ─────────────────────────────────────────────

def save_predictions_with_features(X_test_df, y_test, y_pred, target_col):
    """Save feature values + actual + predicted to CSV."""
    out = X_test_df.copy().reset_index(drop=True)
    out[f"actual_{target_col}"]    = y_test
    out[f"predicted_{target_col}"] = y_pred
    fname = f"knn_{target_col}_predictions_full.csv"
    out.to_csv(fname, index=False)
    print(f"  Saved {fname}")
    return out


# ─────────────────────────────────────────────
# VISUALIZATIONS (Altair)
# ─────────────────────────────────────────────

# Human-readable label maps — update codes to match your BRFSS codebook
EDUCATION_LABELS = {
    1: "Never attended", 2: "Grades 1-8", 3: "Grades 9-11",
    4: "Grade 12/GED",   5: "Some college", 6: "College grad",
}
INCOME_LABELS = {
    1: "<$10k",    2: "$10-15k",  3: "$15-20k",   4: "$20-25k",
    5: "$25-35k",  6: "$35-50k",  7: "$50-75k",   8: "$75-100k",
    9: "$100-150k", 10: "$150-200k", 11: ">$200k",
}
SEX_LABELS        = {1: "Male", 2: "Female"}
EMPLOYMENT_LABELS = {
    1: "Employed wages", 2: "Self-employed",   3: "Out of work >1yr",
    4: "Out of work <1yr", 5: "Homemaker",     6: "Student",
    7: "Retired",          8: "Unable to work",
}
INSURANCE_LABELS  = {1: "Yes, one plan", 2: "Yes, multiple", 3: "No insurance"}

TARGET_CLASS_LABELS = {
    "diabetes":     {0: "Not diabetic", 1: "Diabetic"},
    "hypertension": {1: "Has condition", 2: "No condition"},
    "cholesterol":  {1: "Diagnosed", 2: "Not diagnosed"},
}


def decode_ohe(df, prefix, label_map):
    """
    Collapse one-hot encoded columns back into a single readable string column.
    e.g. employment_1.0, employment_2.0 → "Employed wages", "Self-employed", etc.
    """
    ohe_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if not ohe_cols:
        return df

    def get_label(row):
        for c in ohe_cols:
            if row[c] == 1:
                try:
                    code = int(float(c.split("_")[-1]))
                    return label_map.get(code, str(code))
                except ValueError:
                    return "Unknown"
        return "Unknown"

    df[f"{prefix}_label"] = df.apply(get_label, axis=1)
    return df


def prep_predictions(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Add readable label columns to a predictions DataFrame."""
    df = df.copy()
    pred_col = f"predicted_{target}"

    df["predicted_label"] = df[pred_col].map(
        TARGET_CLASS_LABELS.get(target, {})).fillna(df[pred_col].astype(str))

    if "education" in df.columns:
        df["education_label"] = df["education"].map(EDUCATION_LABELS)
        df["education_order"] = df["education"]
    if "income" in df.columns:
        df["income_label"] = df["income"].map(INCOME_LABELS)
        df["income_order"] = df["income"]
    if "sex" in df.columns:
        df["sex_label"] = df["sex"].map(SEX_LABELS)
    if "age" in df.columns:
        bins   = [17, 24, 34, 44, 54, 64, 74, 80]
        labels = ["18-24","25-34","35-44","45-54","55-64","65-74","75-80"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels).astype(str)

    df = decode_ohe(df, "employment", EMPLOYMENT_LABELS)
    df = decode_ohe(df, "insurance",  INSURANCE_LABELS)

    return df


def make_proportion_chart(df: pd.DataFrame, x_col: str, x_title: str,
                          target: str, sort_col: str = None) -> alt.Chart:
    """
    100% stacked bar chart: proportion of each predicted class per category.
    Click legend to highlight. Hover for exact counts and percentages.
    """
    pred_col = "predicted_label"
    counts   = (df.groupby([x_col, pred_col])
                  .size()
                  .reset_index(name="count"))
    totals              = counts.groupby(x_col)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    counts["pct_text"]   = (counts["proportion"] * 100).round(1).astype(str) + "%"

    if sort_col and sort_col in df.columns:
        order_map      = df[[x_col, sort_col]].drop_duplicates().set_index(x_col)[sort_col]
        counts["_sort"] = counts[x_col].map(order_map)
        x_enc = alt.X(f"{x_col}:N", title=x_title,
                      sort=alt.EncodingSortField(field="_sort", order="ascending"))
    else:
        x_enc = alt.X(f"{x_col}:N", title=x_title, sort="ascending")

    class_labels = list(TARGET_CLASS_LABELS.get(target, {}).values())
    color_scale  = alt.Scale(domain=class_labels,
                             range=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    selection    = alt.selection_point(fields=[pred_col], bind="legend")

    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=x_enc,
            y=alt.Y("proportion:Q", title="Proportion",
                    axis=alt.Axis(format="%")),
            color=alt.Color(f"{pred_col}:N", title="Predicted class",
                            scale=color_scale),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[
                alt.Tooltip(f"{x_col}:N",         title=x_title),
                alt.Tooltip(f"{pred_col}:N",       title="Predicted"),
                alt.Tooltip("proportion:Q",        title="Proportion", format=".1%"),
                alt.Tooltip("count:Q",             title="Count"),
            ]
        )
        .add_params(selection)
        .properties(width=440, height=280,
                    title=f"Predicted {target.capitalize()} by {x_title}")
    )


def build_target_charts(df: pd.DataFrame, target: str) -> alt.VConcatChart:
    """Build all charts for one health outcome."""
    df = prep_predictions(df, target)
    charts = []

    if "education_label" in df.columns:
        charts.append(make_proportion_chart(
            df, "education_label", "Education Level", target, "education_order"))
    if "income_label" in df.columns:
        charts.append(make_proportion_chart(
            df, "income_label", "Household Income", target, "income_order"))
    if "age_group" in df.columns:
        charts.append(make_proportion_chart(
            df, "age_group", "Age Group", target))
    if "sex_label" in df.columns:
        charts.append(make_proportion_chart(
            df, "sex_label", "Sex", target))
    if "employment_label" in df.columns:
        charts.append(make_proportion_chart(
            df, "employment_label", "Employment Status", target))
    if "insurance_label" in df.columns:
        charts.append(make_proportion_chart(
            df, "insurance_label", "Insurance Coverage", target))

    # Arrange into rows of 2
    rows = []
    for i in range(0, len(charts), 2):
        pair = charts[i:i+2]
        rows.append(alt.hconcat(*pair).resolve_scale(color="shared"))

    return alt.vconcat(*rows).properties(title=f"── {target.upper()} ──")


def build_dashboard(predictions_dict: dict, output_file="brfss_dashboard.html"):
    """Build and save one HTML dashboard for all health outcomes."""
    alt.data_transformers.enable("default", max_rows=None)

    pages = [build_target_charts(df, target)
             for target, df in predictions_dict.items()
             if df is not None and not df.empty]

    dashboard = (
        alt.vconcat(*pages)
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_title(fontSize=16, anchor="start")
    )

    dashboard.save(output_file)
    print(f"\nDashboard saved → {output_file}  (open in any browser)")


# ─────────────────────────────────────────────
# FULL PIPELINE FOR ONE TARGET
# ─────────────────────────────────────────────

def run_knn_for_target(df, target_col, k_values, max_rows=10000):
    print(f"\n{'=' * 50}")
    print(f"RUNNING KNN FOR: {target_col.upper()}")
    print(f"{'=' * 50}")

    X, y = prepare_features_and_target(df, target_col)
    print("Features:", X.columns.tolist())
    print("Target classes:", sorted(y.unique()))

    X, y = maybe_sample_data(X, y, max_rows=max_rows)
    print("Shape after sampling:", X.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(X, y)
    X_tr_s, X_va_s, X_te_s = scale_datasets(X_train, X_val, X_test)

    y_tr = y_train.to_numpy()
    y_va = y_val.to_numpy()
    y_te = y_test.to_numpy()

    results_df, best_k = test_k_values(X_tr_s, y_tr, X_va_s, y_va, k_values)

    y_pred = predict_all(X_tr_s, y_tr, X_te_s, best_k)
    acc, pre, rec, f1 = evaluate_model(y_te, y_pred)

    print(f"\nBest k: {best_k}")
    print(f"Test  Accuracy  : {acc:.4f}")
    print(f"      Precision : {pre:.4f}")
    print(f"      Recall    : {rec:.4f}")
    print(f"      F1        : {f1:.4f}")

    results_df.to_csv(f"knn_{target_col}_k_results.csv", index=False)

    # Save predictions using unscaled X_test so charts show real feature values
    X_test_df = X_test.reset_index(drop=True)
    save_predictions_with_features(X_test_df, y_te, y_pred, target_col)

    return {
        "target": target_col, "best_k": best_k,
        "accuracy": acc, "precision_weighted": pre,
        "recall_weighted": rec, "f1_weighted": f1,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    use_clean_file = False  # set to True after first successful run
    raw_file   = "brfss_survey_data_2024.csv"
    clean_file = "clean_brfss_data.csv"

    if use_clean_file:
        try:
            df = load_clean_data(clean_file)
            print(f"Loaded cleaned file: {clean_file}  shape={df.shape}")
        except FileNotFoundError:
            print("Cleaned file not found — building from raw data...")
            raw_df = load_data(raw_file)
            col_map = build_column_map(raw_df)
            df = clean_brfss_data(raw_df, col_map)
            save_clean_model_file(df, clean_file)
    else:
        raw_df  = load_data(raw_file)
        col_map = build_column_map(raw_df)
        df      = clean_brfss_data(raw_df, col_map)
        save_clean_model_file(df, clean_file)

    print("Dataset shape:", df.shape)

    targets  = ["diabetes", "hypertension", "cholesterol"]
    k_values = [1, 3, 5, 7, 9, 11]
    results  = []

    for target in targets:
        if target in df.columns:
            results.append(run_knn_for_target(df, target, k_values, max_rows=10000))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("knn_all_health_outcomes_summary.csv", index=False)
    print(f"\n{'=' * 50}\nFINAL SUMMARY\n{'=' * 50}")
    print(summary_df)

    # Build Altair dashboard from saved prediction CSVs
    preds_dict = {}
    for target in targets:
        fname = f"knn_{target}_predictions_full.csv"
        if os.path.exists(fname):
            preds_dict[target] = pd.read_csv(fname)

    build_dashboard(preds_dict, "brfss_dashboard.html")


if __name__ == "__main__":
    main()