# src/api/routes/train.py
from fastapi import APIRouter, HTTPException
import os, json
import pandas as pd
import mlflow

# Local pipeline imports
from aurix_data_cleaner import clean_telco_dataset
from aurix_feature_engineering import engineer_features
from aurix_model_train_xgb import main as train_xgb_main

router = APIRouter(prefix="/train", tags=["train"])

DATA_DIR = os.getenv("AURIX_DATA_DIR", "data")
RAW_XLSX = os.path.join(DATA_DIR, "Telco_customer_churn.xlsx")
CLEAN_CSV = os.path.join(DATA_DIR, "Telco_clean.csv")
FEAT_CSV = os.path.join(DATA_DIR, "aurix_features.csv")
MODEL_PATH = os.getenv("AURIX_MODEL_PATH", "aurix_churn_model_xgb.joblib")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "aurix-ai"))

@router.post("", summary="Retrain churn and uplift models")
def trigger_training():
    try:
        # 1) Clean raw dataset if available
        if os.path.exists(RAW_XLSX):
            clean_telco_dataset(RAW_XLSX, output_path=CLEAN_CSV)
        elif not os.path.exists(CLEAN_CSV):
            raise HTTPException(status_code=400, detail=f"Missing dataset: {RAW_XLSX} or {CLEAN_CSV}")

        # 2) Feature engineering
        df_clean = pd.read_csv(CLEAN_CSV)
        _, _ = engineer_features(df_clean)
        if os.path.exists("aurix_features.csv"):
            os.replace("aurix_features.csv", FEAT_CSV)

        # 3) Train XGB models
        train_xgb_main()

        # 4) Log to MLflow
        metrics_path = "aurix_churn_eval_xgb.json"
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

        with mlflow.start_run(run_name="train_xgb") as run:
            mlflow.log_param("model_type", "XGBClassifier")
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass
            for art in [
                MODEL_PATH,
                "xgb_roc.pdf",
                "xgb_profit_curve.pdf",
                "xgb_feature_importance.pdf",
                "qini_placeholder.pdf",
                "aurix_uplift_model_treated_xgb.joblib",
                "aurix_uplift_model_control_xgb.joblib",
                "aurix_uplift_eval_xgb.csv",
            ]:
                if os.path.exists(art):
                    mlflow.log_artifact(art)

        artifacts = [p for p in [
            MODEL_PATH,
            "aurix_uplift_model_treated_xgb.joblib",
            "aurix_uplift_model_control_xgb.joblib",
            metrics_path,
            FEAT_CSV,
        ] if os.path.exists(p)]
        return {"status": "trained", "artifacts": artifacts, "metrics_path": metrics_path}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
