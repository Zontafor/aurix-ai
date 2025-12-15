# # src/api/routes/predict.py
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import os
# import joblib
# import pandas as pd

# router = APIRouter(prefix="/predict", tags=["predict"])

# MODEL_PATH = os.getenv("AURIX_MODEL_PATH", "aurix_churn_model_xgb.joblib")
# UP_T_PATH = os.getenv("AURIX_UP_TREATED", "aurix_uplift_model_treated_xgb.joblib")
# UP_C_PATH = os.getenv("AURIX_UP_CONTROL", "aurix_uplift_model_control_xgb.joblib")
# FEAT_CSV = os.path.join(os.getenv("AURIX_DATA_DIR", "data"), "aurix_features.csv")

# class PredictRequest(BaseModel):
#     records: List[dict]

# @router.post("", summary="Predict churn and uplift")
# def predict(req: PredictRequest):
#     try:
#         if not os.path.exists(MODEL_PATH):
#             raise HTTPException(status_code=400, detail=f"Model not found: {MODEL_PATH}. Train first.")
#         model = joblib.load(MODEL_PATH)

#         uplift_t = joblib.load(UP_T_PATH) if os.path.exists(UP_T_PATH) else None
#         uplift_c = joblib.load(UP_C_PATH) if os.path.exists(UP_C_PATH) else None

#         X = pd.DataFrame(req.records)

#         # Align to feature set if available
#         feat_df = None
#         if os.path.exists(FEAT_CSV):
#             feat_df = pd.read_csv(FEAT_CSV, nrows=1)
#             common = [c for c in X.columns if c in feat_df.columns and c != "churn_flag"]
#             X = X[common]
#             for col in feat_df.columns:
#                 if col == "churn_flag":
#                     continue
#                 if col not in X.columns:
#                     X[col] = 0
#             X = X[[c for c in feat_df.columns if c != "churn_flag"]]

#         churn_prob = model.predict_proba(X)[:, 1].tolist()
#         uplift = None
#         if uplift_t is not None and uplift_c is not None:
#             p_t = uplift_t.predict_proba(X.to_numpy())[:, 1]
#             p_c = uplift_c.predict_proba(X.to_numpy())[:, 1]
#             uplift = (p_t - p_c).tolist()

#         return {"n": len(churn_prob), "churn_prob": churn_prob, "uplift": uplift}
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # src/api/predict.py
# # AURIX-AI: churn risk prediction endpoint.

# from pathlib import Path
# from typing import Optional

# import joblib
# import pandas as pd
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel

# router = APIRouter(tags=["prediction"])

# # Pydantic schemas (shared with /recommend_action)
# class CustomerFeatures(BaseModel):
#     """
#     Minimal customer feature schema used for online scoring.

#     This is aligned with the cleaned Telco schema (Telco_clean.csv) and
#     is sufficient to demonstrate a working FastAPI ML service.
#     """

#     customerID: str

#     # Core numeric drivers
#     tenure: float
#     MonthlyCharges: float
#     TotalCharges: float

#     # Contract / billing options
#     Contract: str
#     InternetService: str
#     PaymentMethod: str
#     PaperlessBilling: str

#     # Demographics and service flags
#     gender: str
#     SeniorCitizen: int
#     Partner: str
#     Dependents: str
#     PhoneService: str
#     MultipleLines: Optional[str] = None
#     OnlineSecurity: Optional[str] = None
#     OnlineBackup: Optional[str] = None
#     DeviceProtection: Optional[str] = None
#     TechSupport: Optional[str] = None
#     StreamingTV: Optional[str] = None
#     StreamingMovies: Optional[str] = None


# class RiskResponse(BaseModel):
#     customer_id: str
#     churn_probability: float
#     model_version: str


# # Model loading and preprocessing helpers
# _MODEL = None
# _MODEL_VERSION = "xgb_telco_v1"
# _FEATURE_COLUMNS = None


# def _load_model_once() -> None:
#     """Lazy-load the trained XGBoost model (and feature columns, if saved)."""
#     global _MODEL, _FEATURE_COLUMNS
#     if _MODEL is not None:
#         return

#     # src/api/predict.py -> src/api -> src -> project root
#     root = Path(__file__).resolve().parents[2]
#     # model_path = root / "data" / "models" / "aurix_churn_model_xgb.joblib"
#     model_path = root / "aurix_churn_model_xgb.joblib"

#     if not model_path.exists():
#         raise RuntimeError(f"Churn model not found at {model_path}")

#     _MODEL = joblib.load(model_path)
#     _FEATURE_COLUMNS = getattr(_MODEL, "feature_names_in_", None)

# def _align_columns(df: pd.DataFrame) -> pd.DataFrame:
#     if _FEATURE_COLUMNS is None:
#         return df

#     for col in _FEATURE_COLUMNS:
#         if col not in df.columns:
#             df[col] = 0

#     # Any extra columns in df that the model didn't expect are dropped
#     return df.loc[:, _FEATURE_COLUMNS]

# _BOOL_YES_NO = [
#     "PaperlessBilling",
#     "Partner",
#     "Dependents",
#     "PhoneService",
# ]


# def _to_dataframe(payload: CustomerFeatures) -> tuple[str, pd.DataFrame]:
#     """
#     Convert incoming features to a single-row pandas DataFrame and
#     apply lightweight cleaning consistent with the training pipeline.
#     """
#     data = payload.dict()
#     customer_id = data.pop("customerID")

#     df = pd.DataFrame([data])

#     # Map obvious Yes/No flags to 1/0 when present
#     for col in _BOOL_YES_NO:
#         if col in df.columns:
#             df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

#     # Coerce numerics
#     for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

#     return customer_id, df


# def _align_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Align incoming features to the model's expected columns, filling
#     missing ones with zeros.
#     """
#     if _FEATURE_COLUMNS is None:
#         # No explicit feature list stored; just pass through.
#         return df

#     for col in _FEATURE_COLUMNS:
#         if col not in df.columns:
#             df[col] = 0

#     return df[_FEATURE_COLUMNS]


# # Endpoints
# @router.post("/predict_risk", response_model=RiskResponse)
# def predict_risk(features: CustomerFeatures) -> RiskResponse:
#     """
#     Predict churn probability for a single Telco customer.

#     This is the main endpoint used in the paper and HW3 write-up.
#     """
#     try:
#         _load_model_once()
#     except RuntimeError as exc:
#         raise HTTPException(status_code=500, detail=str(exc)) from exc

#     customer_id, df = _to_dataframe(features)
#     X = _align_columns(df)

#     try:
#         proba = float(_MODEL.predict_proba(X)[:, 1][0])
#     except Exception as exc:  # defensive
#         raise HTTPException(
#             status_code=500,
#             detail=f"Model prediction failed: {exc}",
#         ) from exc

#     return RiskResponse(
#         customer_id=customer_id,
#         churn_probability=proba,
#         model_version=_MODEL_VERSION,
#     )


# @router.post("/predict", response_model=RiskResponse, include_in_schema=False)
# def predict_legacy(features: CustomerFeatures) -> RiskResponse:
#     """
#     Backwards-compatible endpoint for old clients that still POST to /predict.
#     Delegates to /predict_risk.
#     """
#     return predict_risk(features)

# src/api/predict.py
# AURIX-AI: churn risk prediction endpoint.

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["prediction"])

# -----------------------------
# Schemas
# -----------------------------
class CustomerFeatures(BaseModel):
    customerID: str

    tenure: float
    MonthlyCharges: float
    TotalCharges: float

    Contract: str
    InternetService: str
    PaymentMethod: str
    PaperlessBilling: str

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str

    MultipleLines: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None


class RiskResponse(BaseModel):
    customer_id: str
    churn_probability: float
    model_version: str


# -----------------------------
# Globals
# -----------------------------
_MODEL = None
_MODEL_VERSION = "xgb_telco_v1"
_FEATURE_COLUMNS: Optional[list[str]] = None

# columns that are Yes/No in this dataset (we convert to 1/0)
_BOOL_YES_NO = ["PaperlessBilling", "Partner", "Dependents", "PhoneService"]

# optional service columns that should default sensibly
_OPTIONAL_SERVICE_COLS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def _load_model_once() -> None:
    """Lazy-load the trained model and its expected feature columns."""
    global _MODEL, _FEATURE_COLUMNS

    if _MODEL is not None:
        return

    root = Path(__file__).resolve().parents[2]
    model_path = root / "aurix_churn_model_xgb.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Churn model not found at {model_path}")

    _MODEL = joblib.load(model_path)

    cols = getattr(_MODEL, "feature_names_in_", None)
    _FEATURE_COLUMNS = list(cols) if cols is not None else None


def _to_dataframe(payload: CustomerFeatures) -> Tuple[str, pd.DataFrame]:
    """Convert request payload to a single-row DataFrame with light cleaning."""
    data = payload.model_dump()
    customer_id = data.pop("customerID")

    df = pd.DataFrame([data])

    # Fill optional service fields so get_dummies doesn't produce NaN behavior
    for c in _OPTIONAL_SERVICE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("No")

    # Map Yes/No flags to 1/0
    for col in _BOOL_YES_NO:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Coerce numeric fields
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return customer_id, df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply simple one-hot encoding to object columns, then align exactly to the
    model's expected columns using a single fast reindex (no fragmentation).
    """
    # One-hot encode any remaining object/string columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df = pd.get_dummies(df, columns=list(obj_cols), dummy_na=False)

    # If the model has an expected column order/list, align in one shot
    if _FEATURE_COLUMNS is not None:
        df = df.reindex(columns=_FEATURE_COLUMNS, fill_value=0)

    return df


@router.post("/predict_risk", response_model=RiskResponse)
def predict_risk(features: CustomerFeatures) -> RiskResponse:
    try:
        _load_model_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    customer_id, df = _to_dataframe(features)
    X = _prepare_features(df)

    try:
        proba = float(_MODEL.predict_proba(X)[:, 1][0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc

    return RiskResponse(
        customer_id=customer_id,
        churn_probability=proba,
        model_version=_MODEL_VERSION,
    )


@router.post("/predict", response_model=RiskResponse, include_in_schema=False)
def predict_legacy(features: CustomerFeatures) -> RiskResponse:
    return predict_risk(features)