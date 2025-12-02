# src/api/routes/explain.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import joblib
import pandas as pd
import numpy as np
import shap

router = APIRouter(prefix="/explain", tags=["explain"])

MODEL_PATH = os.getenv("AURIX_MODEL_PATH", "aurix_churn_model_xgb.joblib")
FEAT_CSV = os.path.join(os.getenv("AURIX_DATA_DIR", "data"), "aurix_features.csv")

class ExplainRequest(BaseModel):
    records: List[dict]
    return_shap_values: bool = False
    max_features: int = 25

@router.post("", summary="SHAP-based local explanations for churn probability")
def explain(req: ExplainRequest):
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail=f"Model not found: {MODEL_PATH}. Train first.")
        model = joblib.load(MODEL_PATH)

        X = pd.DataFrame(req.records)
        # Align to feature set if available
        if os.path.exists(FEAT_CSV):
            feat_cols = pd.read_csv(FEAT_CSV, nrows=1).drop(columns=["churn_flag"], errors="ignore").columns.tolist()
            for col in feat_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[feat_cols]

        # TreeExplainer for XGB
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):  # xgboost may return [class0, class1]
            # Use the positive class explanations
            shap_values = shap_values[1]
        shap_values = np.array(shap_values)

        # Global importance: mean |SHAP|
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        feat_names = X.columns.to_numpy()
        order = np.argsort(-mean_abs)[: int(req.max_features)]
        ranked = [{"feature": feat_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order]

        response = {
            "n": int(len(X)),
            "feature_ranking": ranked,
        }

        if req.return_shap_values:
            response["shap_values"] = shap_values[:, order].tolist()
            response["features"] = feat_names[order].tolist()

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
