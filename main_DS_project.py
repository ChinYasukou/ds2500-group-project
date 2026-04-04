
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(file):
    """
    Load raw BRFSS CSV file.
    """
    return pd.read_csv(file, sep=",", skiprows=1, low_memory=False)


def load_clean_data(file):
    """
    Load already-cleaned CSV file.
    """
    return pd.read_csv(file, low_memory=False)


def find_existing_column(df, possible_names):
    """
    Return the first column name from possible_names that exists in df.
    """
    for col in possible_names:
        if col in df.columns:
            return col
    return None


def build_column_map(df):
    """
    Build a column map using likely BRFSS variable names.
    """
    return {
        "income": find_existing_column(df, ["INCOME3"]),
        "education": find_existing_column(df, ["EDUCA"]),
        "employment": find_existing_column(df, ["EMPLOY1"]),
        "insurance": find_existing_column(df, ["PERSDOC3"]),
        "diabetes": find_existing_column(df, ["DIABETE4"]),
        "hypertension": find_existing_column(df, ["_MICHD"]),
        "cholesterol": find_existing_column(df, ["CHCSCNC1"]),
        "age": find_existing_column(df, ["_AGE80"]),
        "sex": find_existing_column(df, ["SEXVAR"])
    }


def clean_brfss_data(df, column_map):
    """
    Select relevant columns and replace BRFSS missing-value codes with NaN.
    """
    selected = {k: v for k, v in column_map.items() if v is not None}

    clean_df = df[list(selected.values())].copy()
    clean_df.rename(columns={v: k for k, v in selected.items()}, inplace=True)

    for col in clean_df.columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    missing_codes = {
        7: np.nan, 9: np.nan,
        77: np.nan, 88: np.nan, 99: np.nan,
        777: np.nan, 888: np.nan, 999: np.nan,
        7777: np.nan, 8888: np.nan, 9999: np.nan
    }

    clean_df.replace(missing_codes, inplace=True)
    return clean_df


def fill_missing_values(clean_df):
    """
    Fill missing values with the mode for each column.
    """
    filled_df = clean_df.copy()

    for col in filled_df.columns:
        if filled_df[col].isna().sum() > 0:
            mode_value = filled_df[col].mode(dropna=True)
            if len(mode_value) > 0:
                filled_df[col] = filled_df[col].fillna(mode_value.iloc[0])
                
    return filled_df


def save_clean_model_file(df, file_name="clean_brfss_data.csv"):
    """
    Save cleaned dataframe to CSV.
    """
    df.to_csv(file_name, index=False)
    print(f"\nSaved cleaned data as {file_name}")

# this is where stats and graphs for coorelation go

def prepare_features_and_target(df, target_col):
    """
    Select predictor variables and one target variable.
    """
    feature_cols = ["income", "education", "employment", "insurance", "age", "sex"]
    available_features = [col for col in feature_cols if col in df.columns]

    needed_cols = available_features + [target_col]
    model_df = df[needed_cols].dropna().copy()

    X = model_df[available_features]
    y = model_df[target_col]

    return X, y


def maybe_sample_data(X, y, max_rows=None, random_state=42):
    """
    Optionally sample the dataset to speed up KNN.
    """
    if max_rows is None or len(X) <= max_rows:
        return X, y

    sampled_df = X.copy()
    sampled_df["target"] = y.values
    sampled_df = sampled_df.sample(n=max_rows, random_state=random_state)

    y_sampled = sampled_df["target"].copy()
    X_sampled = sampled_df.drop(columns=["target"])

    return X_sampled, y_sampled


def split_train_validation_test(X, y, train_size=0.6, val_size=0.2,
                                test_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - train_size),
        random_state=random_state,
        stratify=y
    )

    relative_test_size = test_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_datasets(X_train, X_val, X_test):
    """
    Standardize features using the training set only.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def compute_distances_vectorized(X_train, test_row):
    """
    Compute Euclidean distances from one test row to all training rows.
    """
    return np.sqrt(np.sum((X_train - test_row) ** 2, axis=1))


def predict_one(X_train, y_train, test_row, k):
    """
    Predict one class label using KNN.
    """
    distances = compute_distances_vectorized(X_train, test_row)
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]

    values, counts = np.unique(nearest_labels, return_counts=True)
    prediction = values[np.argmax(counts)]

    return prediction


def predict_all(X_train, y_train, X_test, k):
    """
    Predict class labels for all rows in X_test.
    """
    predictions = []

    for row in X_test:
        pred = predict_one(X_train, y_train, row, k)
        predictions.append(pred)

    return np.array(predictions)


def evaluate_model(y_true, y_pred):
    """
    Compute multiclass evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return accuracy, precision, recall, f1


def test_k_values(X_train, y_train, X_val, y_val, k_values):
    """
    Test several k values on the validation set.
    """
    results = []

    for k in k_values:
        y_val_pred = predict_all(X_train, y_train, X_val, k)
        accuracy, precision, recall, f1 = evaluate_model(y_val, y_val_pred)

        results.append({
            "k": k,
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1
        })

        print(
            f"k={k} | Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}"
        )

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["f1_weighted"].idxmax()]
    best_k = int(best_row["k"])

    return results_df, best_k


def run_knn_for_target(df, target_col, k_values, max_rows=10000):
    """
    Run the full KNN workflow for one target variable.
    """
    print(f"\n{'=' * 50}")
    print(f"RUNNING KNN FOR: {target_col.upper()}")
    print(f"{'=' * 50}")

    X, y = prepare_features_and_target(df, target_col)

    print("Features being used:")
    print(X.columns.tolist())

    print("Target values:")
    print(sorted(y.unique()))

    X, y = maybe_sample_data(X, y, max_rows=max_rows)
    print("Shape after optional sampling:", X.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = scale_datasets(X_train, X_val, X_test)

    y_train_array = y_train.to_numpy()
    y_val_array = y_val.to_numpy()
    y_test_array = y_test.to_numpy()

    results_df, best_k = test_k_values(
        X_train_scaled, y_train_array, X_val_scaled, y_val_array, k_values
    )

    y_test_pred = predict_all(X_train_scaled, y_train_array, X_test_scaled, best_k)
    accuracy, precision, recall, f1 = evaluate_model(y_test_array, y_test_pred)

    print(f"\nBest k for {target_col}: {best_k}")
    print("Test Accuracy:", round(accuracy, 4))
    print("Test Weighted Precision:", round(precision, 4))
    print("Test Weighted Recall:", round(recall, 4))
    print("Test Weighted F1:", round(f1, 4))

    results_df.to_csv(f"knn_{target_col}_k_results.csv", index=False)

    save_predictions_with_features(
    X_test,
    y_test_array,
    y_test_pred,
    X.columns.tolist(),
    target_col)

    summary = {
        "target": target_col,
        "best_k": best_k,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1
    }

    return summary

def save_predictions_with_features(X_test, y_test, y_pred, feature_names, target_col):
    """
    Save a CSV file that includes:
    - socioeconomic features
    - actual values
    - predicted values

    Parameters
    ----------
    X_test : array-like
        Test feature data (scaled or unscaled).
    y_test : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    feature_names : list
        Names of feature columns.
    target_col : str
        Name of the target variable (e.g., 'diabetes').
    """

    # Convert X_test back into a DataFrame
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Create full predictions DataFrame
    predictions_df = X_test_df.copy()
    predictions_df[f"actual_{target_col}"] = y_test
    predictions_df[f"predicted_{target_col}"] = y_pred

    # Save to CSV
    file_name = f"knn_{target_col}_predictions_full.csv"
    predictions_df.to_csv(file_name, index=False)

    print(f"Saved full predictions file: {file_name}")

def main():
    """
    Run KNN for all selected health outcomes.
    """
    use_clean_file = True
    raw_file = "brfss_survey_data_2024.csv"
    clean_file = "clean_brfss_data.csv"

    if use_clean_file:
        try:
            df = load_clean_data(clean_file)
            print(f"Loaded cleaned file: {clean_file}")
        except FileNotFoundError:
            print("Cleaned file not found. Creating it from raw data...")
            raw_df = load_data(raw_file)
            column_map = build_column_map(raw_df)
            clean_df = clean_brfss_data(raw_df, column_map)
            df = fill_missing_values(clean_df)
            save_clean_model_file(df, clean_file)
    else:
        raw_df = load_data(raw_file)
        column_map = build_column_map(raw_df)
        clean_df = clean_brfss_data(raw_df, column_map)
        df = fill_missing_values(clean_df)

    print("\nLoaded dataset shape:", df.shape)

    targets = ["diabetes", "hypertension", "cholesterol"]
    k_values = [1, 3, 5, 7, 9, 11]
    all_results = []

    for target in targets:
        if target in df.columns:
            summary = run_knn_for_target(df, target, k_values, max_rows=10000)
            all_results.append(summary)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("knn_all_health_outcomes_summary.csv", index=False)

    print(f"\n{'=' * 50}")
    print("FINAL SUMMARY")
    print(f"{'=' * 50}")
    print(summary_df)
    

if __name__ == "__main__":
    main()