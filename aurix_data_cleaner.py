import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_dataset(df: pd.DataFrame, figs_dir: Path) -> None:
    """
    Provide structural and statistical inspection of the dataset.
    Includes summary statistics, missing value report, and correlation heatmap.
    Saves output to figs_dir.
    """
    print("\nDataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isna().sum())

    # Numeric summary
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        print("\nNumeric feature summary:\n", num_df.describe())

        # Correlation heatmap
        corr = num_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()

        figs_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = figs_dir / "correlation_heatmap.pdf"
        plt.savefig(heatmap_path)
        plt.close()

        print(f"Saved correlation heatmap to {heatmap_path}.")


def clean_telco_dataset(filepath: str, output_path: Path) -> pd.DataFrame:
    """
    Load, inspect, and clean the Telco Customer Churn dataset.
    Performs schema validation, missing value handling, and categorical encoding.
    """
    df = pd.read_excel(filepath)

    # Ensure figs directory exists
    figs_dir = output_path.parent / "figs"
    inspect_dataset(df, figs_dir)

    # Drop duplicates
    df = df.drop_duplicates()

    # Replace blank strings with NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Fill numeric NaNs with median and categorical NaNs with mode
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical features
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Save cleaned dataset
    df_encoded.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path} with shape {df_encoded.shape}.")

    return df_encoded


if __name__ == "__main__":
    # Base data directory \\ can override with env var
    data_dir = Path(os.getenv("AURIX_DATA_DIR", "data"))
    input_file = data_dir / "Telco_customer_churn.xlsx"

    if not input_file.exists():
        raise FileNotFoundError(f"Telco churn file not found at: {input_file}")

    output_file = data_dir / "Telco_clean.csv"
    clean_telco_dataset(str(input_file), output_file)