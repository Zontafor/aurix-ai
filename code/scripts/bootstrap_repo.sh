#!/usr/bin/env bash
# scripts/bootstrap_repo.sh
# AURIX-AI Bootstrap Script
# Usage:
#   bash scripts/bootstrap_repo.sh
#   bash scripts/bootstrap_repo.sh with-api
#   export DBT_PROFILES_DIR="/Users/mlwu/Documents/Academia/CMU/tepper_courses/Machine Learning for Business Applications/project/code/data/dbt_project/profiles"
# 
# Note:
#   protobuf compatability:
#   `pip install streamlit mlflow-skinny --upgrade --upgrade-strategy eager`
# 
#   futher compatability issues? try the following:
#   ```pip install \
#   "numpy<2,>=1.22" \
#   "pyarrow>=4,<16" \
#   "tenacity>=7,<9" \
#   "mlflow==2.15.0" \
#   "mlflow-skinny==2.15.0" \
#   "protobuf==5.29.5" \
#   "dbt-core==1.9.4" \
#   "dbt-adapters==1.14.6"
#   "dbt-sqlite==1.9.0"
#   --force-reinstall```

set -euo pipefail

# Root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

echo "[aurix] Root directory: ${ROOT_DIR}"
echo "[aurix] PYTHONPATH set to: ${PYTHONPATH}"

# Verify virtual environment exists
if [ ! -d ".venv/aurix-ai" ]; then
  echo "[aurix] ERROR: .venv/aurix-ai not found."
  echo "        Please create and populate the venv, e.g.:"
  echo "          python3.11 -m venv .venv/aurix-ai"
  echo "          source .venv/aurix-ai/bin/activate"
  echo "          pip install -r requirements.txt"
  exit 1
fi

echo "[aurix] Ensure your venv is active:"
echo "  source .venv/aurix-ai/bin/activate"

# Data directory setup
DATA_DIR="${ROOT_DIR}/data"
FIG_DIR="${DATA_DIR}/figs"
DBT_DIR="${DATA_DIR}/dbt_project"
SEED_CSV="${DBT_DIR}/telco_sample.csv"

mkdir -p "${DATA_DIR}"
mkdir -p "${FIG_DIR}"

echo "[aurix] Data directories ensured:"
echo "  ${DATA_DIR}"
echo "  ${FIG_DIR}"

# Ensure expected data files exist
RAW_XLS="${DATA_DIR}/Telco_customer_churn.xlsx"
CLEAN_CSV="${DATA_DIR}/Telco_clean.csv"

if [ ! -f "${RAW_XLS}" ]; then
  echo "[aurix] ERROR: Missing ${RAW_XLS}"
  echo "        Please place Telco_customer_churn.xlsx into: ${DATA_DIR}"
  exit 1
fi

if [ ! -f "${CLEAN_CSV}" ]; then
  echo "[aurix] WARNING: ${CLEAN_CSV} is missing."
  echo "[aurix] Running data cleaner to generate it..."
  python aurix_data_cleaner.py
  echo "[aurix] Telco_clean.csv generated."
else
  echo "[aurix] Found ${CLEAN_CSV}"
fi

# Ensure dbt project directory exists
if [ ! -d "${DBT_DIR}" ]; then
  echo "[aurix] ERROR: dbt project directory not found at:"
  echo "        ${DBT_DIR}"
  echo "        Please ensure data/dbt_project exists with dbt_project.yml."
  exit 1
fi

# Ensure dbt seed CSV (telco_sample.csv)
if [ ! -f "${SEED_CSV}" ]; then
  echo "[aurix] WARNING: telco_sample.csv not found in dbt_project."
  echo "          Please create it from Telco_clean.csv or re-run seeding logic."
else
  echo "[aurix] Found dbt seed CSV: ${SEED_CSV}"
fi

if command -v dbt &>/dev/null; then

  # Determine effective profiles directory
  DBT_PROFILES_DIR_EFFECTIVE=""

  # 1) If user has set DBT_PROFILES_DIR and it exists, respect it
  if [ -n "${DBT_PROFILES_DIR:-}" ] && [ -d "${DBT_PROFILES_DIR}" ]; then
    DBT_PROFILES_DIR_EFFECTIVE="${DBT_PROFILES_DIR}"
    echo "[aurix] Using DBT_PROFILES_DIR from environment:"
    echo "        ${DBT_PROFILES_DIR_EFFECTIVE}"

  # 2) Else, if data/dbt_project/profiles exists, use that
  elif [ -d "${DBT_DIR}/profiles" ]; then
    DBT_PROFILES_DIR_EFFECTIVE="${DBT_DIR}/profiles"
    echo "[aurix] Found local dbt profiles directory:"
    echo "        ${DBT_PROFILES_DIR_EFFECTIVE}"
    export DBT_PROFILES_DIR="${DBT_PROFILES_DIR_EFFECTIVE}"

  # 3) Else, fall back to dbt default (~/.dbt) and warn
  else
    echo "[aurix] WARNING: No explicit dbt profiles directory found."
    echo "[aurix]          dbt will look in its default (~/.dbt)."
    echo "[aurix]          To use a project-local profile, create:"
    echo "              ${DBT_DIR}/profiles/profiles.yml"
    echo "[aurix]          and/or set DBT_PROFILES_DIR accordingly."
  fi

  echo "[aurix] Running dbt seed + dbt run in ${DBT_DIR}"

  (
    # Allow dbt to fail without killing the whole bootstrap
    set +e
    cd "${DBT_DIR}"

    # Build profiles flag only if we have an effective directory
    if [ -n "${DBT_PROFILES_DIR_EFFECTIVE}" ]; then
      PROFILES_FLAG=(--profiles-dir "${DBT_PROFILES_DIR_EFFECTIVE}")
    else
      PROFILES_FLAG=()
    fi

    dbt seed "${PROFILES_FLAG[@]}"
    if [ $? -ne 0 ]; then
      echo "[aurix] WARNING: dbt seed failed (likely missing or misconfigured profiles.yml)."
      echo "[aurix]          Continuing without dbt-generated tables."
    fi

    dbt run "${PROFILES_FLAG[@]}"
    if [ $? -ne 0 ]; then
      echo "[aurix] WARNING: dbt run failed (likely missing or misconfigured profiles.yml)."
      echo "[aurix]          Continuing without dbt-generated tables."
    fi
  )

  echo "[aurix] dbt step completed (with possible warnings)."

else
  echo "[aurix] dbt not found in PATH; skipping dbt step."
  echo "[aurix] To enable dbt, install it in your venv, e.g.:"
  echo "        pip install dbt-core dbt-sqlite"
fi

# Feature engineering
echo "[aurix] Running feature engineering (aurix_feature_engineering.py)..."
python aurix_feature_engineering.py
echo "[aurix] Feature engineering complete."

# Model training (XGBoost + uplift)
echo "[aurix] Running training pipeline (XGBoost + uplift)..."
python aurix_model_train_xgb.py
echo "[aurix] Model training complete."

# Evaluation + metrics report
echo "[aurix] Running evaluation + metrics report..."
python aurix_evaluate.py
echo "[aurix] Evaluation complete. AUC/profit/Qini artifacts should now be available."

# Optional: start FastAPI
if [ "${1:-}" = "with-api" ]; then
  echo "[aurix] Starting FastAPI on http://0.0.0.0:8080"
  echo "[aurix] Press Ctrl+C to stop the server."
  uvicorn src.api.main:app --host 0.0.0.0 --port 8080
else
  echo "[aurix] Bootstrap complete!"
  echo "        To start the API manually, run:"
  echo "          uvicorn src.api.main:app --host 0.0.0.0 --port 8080"
fi


# # dbt step (non-fatal; continues even if dbt is not configured)
# if command -v dbt &>/dev/null; then
#   echo "[aurix] Running dbt seed + dbt run in ${DBT_DIR}"
#   (
#     # Allow dbt to fail without killing the whole bootstrap
#     set +e
#     cd "${DBT_DIR}"
#     dbt seed
#     if [ $? -ne 0 ]; then
#       echo "[aurix] WARNING: dbt seed failed (likely missing or misconfigured profiles.yml)."
#       echo "[aurix]          Continuing without dbt-generated tables."
#     fi

#     dbt run
#     if [ $? -ne 0 ]; then
#       echo "[aurix] WARNING: dbt run failed (likely missing or misconfigured profiles.yml)."
#       echo "[aurix]          Continuing without dbt-generated tables."
#     fi
#   )
#   echo "[aurix] dbt step completed (with possible warnings)."
# else
#   echo "[aurix] dbt not found in PATH; skipping dbt step."
# fi

# # Feature engineering
# echo "[aurix] Running feature engineering (aurix_feature_engineering.py)..."
# python aurix_feature_engineering.py
# echo "[aurix] Feature engineering complete."

# # Model training (XGBoost + uplift)
# echo "[aurix] Running training pipeline (XGBoost + uplift)..."
# python aurix_model_train_xgb.py
# echo "[aurix] Model training complete."

# # Evaluation + metrics report
# echo "[aurix] Running evaluation + metrics report..."
# python aurix_evaluate.py
# echo "[aurix] Evaluation complete. AUC/profit/Qini artifacts should now be available."

# # Optional: start FastAPI
# if [ "${1:-}" = "with-api" ]; then
#   echo "[aurix] Starting FastAPI on http://0.0.0.0:8080"
#   echo "[aurix] Press Ctrl+C to stop the server."
#   uvicorn src.api.main:app --host 0.0.0.0 --port 8080
# else
#   echo "[aurix] Bootstrap complete!"
#   echo "        To start the API manually, run:"
#   echo "          uvicorn src.api.main:app --host 0.0.0.0 --port 8080"
# fi

# # Ensure dbt seed CSV (telco_sample.csv)
# if [ ! -f "${SEED_CSV}" ]; then
#   echo "[aurix] telco_sample.csv not found in dbt_project; creating from Telco_clean.csv..."
#   python - << 'PY'
# import pandas as pd
# from pathlib import Path

# root = Path(".").resolve()
# clean_path = root / "data" / "Telco_clean.csv"
# seed_path = root / "data" / "dbt_project" / "telco_sample.csv"

# df = pd.read_csv(clean_path)

# # Simple 20% stratified sample by churn label if available
# label_col = None
# for cand in ["Churn", "churn", "churn_flag", "churn_label"]:
#     if cand in df.columns:
#         label_col = cand
#         break

# if label_col is not None:
#     # Stratified sample by target
#     sample = (
#         df.groupby(label_col, group_keys=False)
#           .apply(lambda g: g.sample(frac=0.2, random_state=46))
#     )
# else:
#     # Fallback: unstratified sample
#     sample = df.sample(frac=0.2, random_state=46)

# seed_path.parent.mkdir(parents=True, exist_ok=True)
# sample.to_csv(seed_path, index=False)
# print(f"[aurix] Wrote seed CSV to: {seed_path}")
# PY
# else
#   echo "[aurix] Found dbt seed CSV: ${SEED_CSV}"
# fi

# # Run dbt seed + run
# if command -v dbt &>/dev/null; then
#   echo "[aurix] Running dbt seed + dbt run in ${DBT_DIR}"
#   (
#     cd "${DBT_DIR}"
#     # Add profile/target flags here if needed for your adapter
#     dbt seed
#     dbt run
#   )
#   echo "[aurix] dbt models built successfully."
# else
#   echo "[aurix] ERROR: dbt not found in PATH."
#   echo "        Please activate your venv and install dbt-core (+ adapter), e.g.:"
#   echo "          pip install dbt-core dbt-duckdb    # or dbt-sqlite, etc."
#   echo "        Then re-run:"
#   echo "          bash scripts/bootstrap_repo.sh"
#   exit 1
# fi

# # Feature engineering
# echo "[aurix] Running feature engineering (aurix_feature_engineering.py)..."
# python aurix_feature_engineering.py
# echo "[aurix] Feature engineering complete."

# # Model training (XGBoost + uplift)
# echo "[aurix] Running training pipeline (XGBoost + uplift)..."
# python aurix_model_train_xgb.py
# echo "[aurix] Model training complete."

# # Evaluation + metrics report
# echo "[aurix] Running evaluation + metrics report..."
# python aurix_evaluate.py
# echo "[aurix] Evaluation complete. AUC/profit/Qini artifacts should now be available."

# # Optional: start FastAPI
# if [ "${1:-}" = "with-api" ]; then
#   echo "[aurix] Starting FastAPI on http://0.0.0.0:8080"
#   echo "[aurix] Press Ctrl+C to stop the server."
#   uvicorn src.api.main:app --host 0.0.0.0 --port 8080
# else
#   echo "[aurix] Bootstrap complete!"
#   echo "        To start the API manually, run:"
#   echo "          uvicorn src.api.main:app --host 0.0.0.0 --port 8080"
# fi