import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

# Set up data directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = Path("/Users/mlwu/Documents/Academia/CMU/tepper_courses/Machine Learning for Business Applications/project/code/data/figs")
DATA_OUT_DIR = DATA_DIR / "data_out"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR = Path("/Users/mlwu/Documents/Academia/CMU/tepper_courses/Machine Learning for Business Applications/project/code/data/data_out")

# Disable plot titles
plt.rcParams["axes.titlepad"] = 0
plt.rcParams["axes.titlesize"] = 0

# Helper to infer target column from raw dataframe
def _infer_target_column_raw(df: pd.DataFrame) -> Optional[str]:
    """
    Infer a reasonable target column name from the *raw* dataframe
    (before we normalize column names).
    """
    # Exact matches first
    if "Churn" in df.columns:
        return "Churn"
    if "churn" in df.columns:
        return "churn"

    # Look for something like "Churn_Yes" in original column names
    for col in df.columns:
        col_lower = col.lower()
        if "churn" in col_lower and "yes" in col_lower and df[col].nunique() <= 2:
            return col

    # Fallback: any column with "churn" in name and 2 unique values
    for col in df.columns:
        if "churn" in col.lower() and df[col].nunique() <= 2:
            return col

    return None

# Exploratory feature inspection
def inspect_features(
    df: pd.DataFrame,
    figs_dir: Path,
    target_col: Optional[str] = None,
) -> None:
    """
    Provide exploratory visualizations and summaries for key features.
    Saves all figures to figs_dir.
    """
    print(f"\nInspecting dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    figs_dir.mkdir(parents=True, exist_ok=True)

    # Infer target if not specified
    if target_col is None:
        target_col = _infer_target_column_raw(df)
        if target_col is None:
            print("Warning: could not infer a target column for balance plot.")
        else:
            print(f"Inferred target column for plotting: {target_col}")

    # Target distribution
    if target_col is not None and target_col in df.columns:
        plt.figure(figsize=(5, 4))
        df[target_col].value_counts(normalize=True).plot(
            kind="bar",
            color=["steelblue", "salmon"],
        )
        # plt.title(f"Target Balance: {target_col}")
        plt.ylabel("Proportion")
        plt.tight_layout()

        out_path = figs_dir / "target_balance.pdf"
        plt.savefig(out_path)
        plt.close()
        print(f"Target balance chart saved to: {out_path}")
    else:
        print("Target balance plot skipped (no suitable target column found).")

    # # Numeric correlation heatmap
    # num_df = df.select_dtypes(include=[np.number])
    # if not num_df.empty:
    #     corr = num_df.corr()
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(corr, cmap="coolwarm", annot=False)
    #     # plt.title("Numeric Feature Correlation")
    #     plt.tight_layout()

    #     out_path = figs_dir / "feature_corr_heatmap.pdf"
    #     plt.savefig(out_path)
    #     plt.close()
    #     print(f"Correlation heatmap saved to: {out_path}")

        # 2) Correlation heatmap for engineered numeric features (lightweight)
    # Compute corr on numeric columns only; with many OHE cols this is big, so we sample top-variance numeric cols.
    # numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    # if len(numeric_cols) == 0:
    #     print("No numeric columns found for correlation plot; skipping.")
    #     return

    # # Cap at 60 cols to keep plot readable
    # if len(numeric_cols) > 60:
    #     var = X[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
    #     numeric_cols = list(var.head(60).index)

    # corr = X[numeric_cols].corr(numeric_only=True)

    # plt.figure(figsize=(10, 8))
    # plt.imshow(corr.values, aspect="auto")
    # plt.colorbar()
    # plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90, fontsize=6)
    # plt.yticks(range(len(numeric_cols)), numeric_cols, fontsize=6)
    # out2 = figs_dir / "feature_corr_heatmap.pdf"
    # plt.savefig(out2, bbox_inches="tight")
    # plt.close()
    # print(f"Correlation heatmap saved to: {out2}")

 # Define leakage/label columns to exclude from correlation
    leakage_cols = {
        target_col,
        "Churn Label_Yes",
        "Churn Score",
        "Churn Value",
    }

    feature_cols = [c for c in df.columns if c not in leakage_cols]
    X = df[feature_cols]

    # numeric-only correlation
    corr = X.select_dtypes(include=[np.number]).corr()

    if corr.empty:
        print("[aurix] No numeric features found for correlation heatmap; skipping.")
        return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(
    corr.values,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto",
    interpolation="nearest"
    )
    # cax = ax.imshow(corr.values, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=6)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("")

    fig.tight_layout()
    heatmap_path = figs_dir / "feature_corr_heatmap.pdf"
    fig.savefig(heatmap_path)
    plt.close(fig)
    print(f"Correlation heatmap saved to: {heatmap_path}")

# Feature engineering for churn modeling
def engineer_features(df: pd.DataFrame, output_file: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform feature engineering for churn modeling.
    Returns processed feature matrix X and target vector y.
    Saves engineered CSV to output_file.

    This function is robust to:
    - raw 'Churn' column, or
    - dummy-encoded 'Churn_Yes' / 'churn_yes', or
    - an existing 'churn_flag' column.
    """
    # Normalize column names
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # ---- Infer churn target column on normalized names ----
    target_col = None

    if "churn" in df.columns:
        target_col = "churn"
    elif "churn_flag" in df.columns:
        target_col = "churn_flag"
    else:
        # Look for churn_* dummy column, e.g. churn_yes
        churn_like = [c for c in df.columns if c.startswith("churn_")]
        for c in churn_like:
            unique_vals = set(df[c].dropna().unique().tolist())
            if unique_vals.issubset({0, 1}) or unique_vals.issubset({0.0, 1.0}):
                target_col = c
                break

    if target_col is None:
        raise ValueError(
            'Could not infer target column for churn. '
            'Expected one of: "churn", "churn_flag", or a binary column like "churn_yes".'
        )

    # Construct churn_flag consistently
    if target_col == "churn":
        df["churn_flag"] = df["churn"].apply(
            lambda x: 1 if str(x).strip().lower() in ["yes", "true", "1"] else 0
        )
    elif target_col == "churn_flag":
        # Assume already 0/1
        pass
    else:
        # target_col is something like 'churn_yes'
        df["churn_flag"] = df[target_col].astype(int)

    # Convert numeric fields if present
    for col in ["tenure", "monthlycharges", "totalcharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    # Derived ratio feature
    if all(c in df.columns for c in ["totalcharges", "tenure"]):
        df["avg_monthly_value"] = np.where(
            df["tenure"] > 0,
            df["totalcharges"] / df["tenure"],
            df.get("monthlycharges", 0),
        )

    # Binary features: any column with 2 unique values except churn_flag
    binary_like = [
        c for c in df.columns
        if df[c].nunique() == 2 and c not in ["churn_flag"]
    ]
    for col in binary_like:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})

    # Contract encoding
    if "contract" in df.columns:
        df["contract_type"] = df["contract"].astype("category").cat.codes

    # Replace remaining NaNs
    df = df.fillna(0)

    # Build list of columns to drop from X to avoid target leakage
    drop_cols = set(["churn", "churn_flag"])
    drop_cols.update([c for c in df.columns if c.startswith("churn_")])

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["churn_flag"]

    # Save engineered dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    X.join(y).to_csv(output_file, index=False)
    print(f"Engineered dataset saved to: {output_file} ({X.shape[1]} features)")

    return X, y

# Main execution
if __name__ == "__main__":
    base_dir = Path("data")
    input_file = base_dir / "Telco_clean.csv"
    figs_dir = base_dir / "figs"
    output_file = base_dir / "aurix_features.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Cleaned Telco dataset not found at: {input_file}")

    df = pd.read_csv(input_file)
    inspect_features(df, figs_dir)
    X, y = engineer_features(df, output_file)