# src/data/ingest_telco.py
# Ingest the IBM Telco dataset from a local path or URL into data/Telco_customer_churn.xlsx.

from pathlib import Path
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("aurix.ingest")

REQUIRED_COLS = [
    "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"
]


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.error("Missing required columns", extra={"missing": missing})
        raise ValueError(f"Missing required columns: {missing}")


def ingest_excel(src_path: str, dest_dir: str = "data", dest_name: str = "Telco_customer_churn.xlsx") -> str:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(src_path)
    _validate(df)

    out_path = dest / dest_name
    with pd.ExcelWriter(out_path) as xlw:
        df.to_excel(xlw, index=False)
    logger.info("Ingested Telco Excel", extra={"rows": len(df), "path": str(out_path)})
    return str(out_path)


def ingest_csv(src_path: str, dest_dir: str = "data", dest_name: str = "Telco_clean.csv") -> str:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_path)
    try:
        _validate(df)
    except Exception:
        pass

    out_path = dest / dest_name
    df.to_csv(out_path, index=False)
    logger.info("Ingested Telco CSV", extra={"rows": len(df), "path": str(out_path)})
    return str(out_path)
