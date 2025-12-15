# src/api/routes/monitor.py
# Data and model monitoring endpoints (basic drift summary via Evidently).

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except Exception:  # noqa: BLE001
    EVIDENTLY_AVAILABLE = False

router = APIRouter(prefix="/monitor", tags=["monitor"])

DATA_DIR = os.getenv("AURIX_DATA_DIR", "data")
REFERENCE_PATH = os.path.join(DATA_DIR, "aurix_features_reference.csv")
CURRENT_PATH = os.path.join(DATA_DIR, "aurix_features_current.csv")


class DriftRequest(BaseModel):
    records: Optional[List[dict]] = None
    reference_path: Optional[str] = None
    current_path: Optional[str] = None


@router.post("/drift", summary="Compute simple data drift summary using Evidently")
def drift(request: DriftRequest):
    if not EVIDENTLY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Evidently is not installed or failed to import in this environment.",
        )

    ref_path = request.reference_path or REFERENCE_PATH
    cur_path = request.current_path or CURRENT_PATH

    try:
        if not os.path.exists(ref_path):
            raise HTTPException(status_code=400, detail=f"Reference data not found at {ref_path}")

        ref_df = pd.read_csv(ref_path)

        if request.records is not None:
            cur_df = pd.DataFrame(request.records)
        else:
            if not os.path.exists(cur_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Current data not provided and {cur_path} does not exist.",
                )
            cur_df = pd.read_csv(cur_path)

        common_cols = [c for c in ref_df.columns if c in cur_df.columns]
        if not common_cols:
            raise HTTPException(status_code=400, detail="No overlapping columns between reference and current data.")
        ref_df = ref_df[common_cols]
        cur_df = cur_df[common_cols]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        as_dict = report.as_dict()

        metrics = as_dict.get("metrics", [])
        summary = {
            "dataset_drift": None,
            "share_drifted_features": None,
        }
        for m in metrics:
            if m.get("metric") == "DataDriftTable":
                vals = m.get("result", {})
                summary["dataset_drift"] = vals.get("dataset_drift")
                summary["share_drifted_features"] = vals.get("share_drifted_features")
                break

        return {
            "status": "ok",
            "reference_rows": int(len(ref_df)),
            "current_rows": int(len(cur_df)),
            "summary": summary,
            "raw": as_dict,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
