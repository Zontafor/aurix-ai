"""
aurix_telco_parser.py
Diagnostic parser for the raw Telco Customer Churn Excel dataset.

This script performs structural inspection, schema reporting, and
target-column inference consistent with the conventions used in the
Aurix-AI data-cleaning and feature-engineering pipeline.

Note: Churn Reason column is split into multiple dummy columns in
the raw data after cleaning/engineering.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

# Target-column inference (same logic family as engineering pipeline)
def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Infer a churn-related target column from the raw Excel sheet.
    Works for:
        - 'Churn'
        - 'churn'
        - dummy-encoded forms (e.g., 'Churn_Yes', 'churn_yes')
        - any 0/1 binary churn-like flag
    """
    # Exact raw form
    if 'Churn' in df.columns:
        return 'Churn'
    if 'churn' in df.columns:
        return 'churn'

    # Encoded variants (e.g. Churn_Yes)
    for col in df.columns:
        col_lower = col.lower()
        if 'churn' in col_lower and 'yes' in col_lower and df[col].nunique() <= 2:
            return col

    # Any churn-like binary column
    for col in df.columns:
        if 'churn' in col.lower() and df[col].nunique() <= 2:
            return col

    return None


# Main parser
def parse_telco_dataset(filepath: str) -> pd.DataFrame:
    """
    Load and summarize the Telco Customer Churn dataset.
    Prints structural diagnostics and inferred target information.
    Returns the loaded DataFrame.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    excel = pd.ExcelFile(file_path)
    sheet = excel.sheet_names[0]
    df = excel.parse(sheet)

    print(f"\nLoaded dataset from sheet '{sheet}'")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("Column Overview:")
    print("-" * 70)
    for col in df.columns:
        missing = df[col].isna().sum()
        dtype = df[col].dtype
        print(f"{col:<25} | {str(dtype):<12} | Missing: {missing:<6}")

    # Infer target
    target_col = infer_target_column(df)
    if target_col:
        print(f"\nInferred target-like column: {target_col}")
    else:
        print("\nWarning: No churn-like target column was detected.")

    return df

# Main execution
if __name__ == "__main__":
    base_dir = Path("data")
    input_file = base_dir / "Telco_customer_churn.xlsx"

    df = parse_telco_dataset(str(input_file))

# import pandas as pd

# def parse_telco_dataset(filepath: str) -> pd.DataFrame:
#     """
#     Load and summarize the Telco Customer Churn dataset.
#     Returns the DataFrame and prints dataset diagnostics.
#     """
#     excel_file = pd.ExcelFile(filepath)
#     df = excel_file.parse(excel_file.sheet_names[0])

#     print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {excel_file.sheet_names[0]}.")
#     print("\nColumn overview:")
#     for col in df.columns:
#         missing = df[col].isna().sum()
#         dtype = df[col].dtype
#         print(f"  {col:<25} | {str(dtype):<10} | Missing: {missing:<5}")

#     return df

# if __name__ == "__main__":
#     df = parse_telco_dataset("Telco_customer_churn.xlsx")