import pandas as pd
import numpy as np
from scipy import stats

# Data Quality Validation
def validate_data(df):
    if df.isnull().values.any():
        print("Data contains missing values.")
    if (df.dtypes == 'object').any() and not df.select_dtypes(include=['object']).apply(lambda x: x.str.isalnum()).all().all():
        print("Data contains non-alphanumeric values.")

# Outlier Detection using Z-score
def detect_outliers_z(df, threshold=3):
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[col]))
        outlier_indices = np.where(z_scores > threshold)
        outliers[col] = df.iloc[outlier_indices]
    return outliers

# Normality Test
def normality_test(df, alpha=0.05):
    results = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stat, p = stats.shapiro(df[col])  # Shapiro-Wilk test
        results[col] = (stat, p)
        if p < alpha:
            print(f"{col} is not normally distributed (p-value: {p})")
        else:
            print(f"{col} is normally distributed (p-value: {p})")
    return results

# Data Integrity Checks
def check_data_integrity(df, expected_columns):
    if set(expected_columns).issubset(df.columns):
        print("Data integrity: All expected columns are present.")
    else:
        missing_cols = set(expected_columns) - set(df.columns)
        print(f"Data integrity: Missing columns {missing_cols}.")

# Example usage:
if __name__ == "__main__":
    # Load your data here
    # df = pd.read_csv('your_data.csv')
    # validate_data(df)
    # outliers = detect_outliers_z(df)
    # results = normality_test(df)
    # check_data_integrity(df, ['col1', 'col2', 'col3'])  # replace with your expected columns
