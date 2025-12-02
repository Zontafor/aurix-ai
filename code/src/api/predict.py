# src/api/routes/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import joblib
import pandas as pd

router = APIRouter(prefix="/predict", tags=["predict"])

MODEL_PATH = os.getenv("AURIX_MODEL_PATH", "aurix_churn_model_xgb.joblib")
UP_T_PATH = os.getenv("AURIX_UP_TREATED", "aurix_uplift_model_treated_xgb.joblib")
UP_C_PATH = os.getenv("AURIX_UP_CONTROL", "aurix_uplift_model_control_xgb.joblib")
FEAT_CSV = os.path.join(os.getenv("AURIX_DATA_DIR", "data"), "aurix_features.csv")

class PredictRequest(BaseModel):
    records: List[dict]

@router.post("", summary="Predict churn and uplift")
def predict(req: PredictRequest):
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail=f"Model not found: {MODEL_PATH}. Train first.")
        model = joblib.load(MODEL_PATH)

        uplift_t = joblib.load(UP_T_PATH) if os.path.exists(UP_T_PATH) else None
        uplift_c = joblib.load(UP_C_PATH) if os.path.exists(UP_C_PATH) else None

        X = pd.DataFrame(req.records)

        # Align to feature set if available
        feat_df = None
        if os.path.exists(FEAT_CSV):
            feat_df = pd.read_csv(FEAT_CSV, nrows=1)
            common = [c for c in X.columns if c in feat_df.columns and c != "churn_flag"]
            X = X[common]
            for col in feat_df.columns:
                if col == "churn_flag":
                    continue
                if col not in X.columns:
                    X[col] = 0
            X = X[[c for c in feat_df.columns if c != "churn_flag"]]

        churn_prob = model.predict_proba(X)[:, 1].tolist()
        uplift = None
        if uplift_t is not None and uplift_c is not None:
            p_t = uplift_t.predict_proba(X.to_numpy())[:, 1]
            p_c = uplift_c.predict_proba(X.to_numpy())[:, 1]
            uplift = (p_t - p_c).tolist()

        return {"n": len(churn_prob), "churn_prob": churn_prob, "uplift": uplift}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
