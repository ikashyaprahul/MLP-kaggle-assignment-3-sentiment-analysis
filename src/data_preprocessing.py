import pandas as pd
import numpy as np


def load_data(train_path, test_path, submission_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    submission_df = pd.read_csv(submission_path)
    return train_df, test_df, submission_df


def preprocess_data(df, text_col, target_col=None):
    # Handle missing values
    df[text_col] = df[text_col].fillna("")
    if target_col:
        df = df.dropna(subset=[target_col])

    # Drop duplicates
    df = df.drop_duplicates()

    # Feature engineering
    df["review_length_chars"] = df[text_col].apply(len)
    df["review_length_words"] = df[text_col].apply(lambda x: len(str(x).split()))

    # Outlier handling
    for col in ["review_length_chars", "review_length_words"]:
        lb, ub = detect_outliers_iqr(df[col])
        df[col] = np.clip(df[col], lb, ub)

    return df


def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound
