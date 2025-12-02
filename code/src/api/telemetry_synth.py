# src/data/telemetry_synth.py
# Synthetic telemetry generator for churn demos (web/session/product events).

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from ..utils.logger import get_logger
logger = get_logger("aurix.telemetry")


def synthesize_sessions(
    n_customers: int = 5000,
    days: int = 120,
    seed: int = 42,
    out_path: str = "data/telemetry_sessions.csv",
) -> str:
    rng = np.random.default_rng(seed)
    start = datetime.utcnow() - timedelta(days=days)
    customers = np.arange(1, n_customers + 1)

    rows = []
    for cid in customers:
        base_rate = rng.uniform(0.05, 0.6)
        user_days = rng.poisson(lam=base_rate * days)
        ts = sorted([start + timedelta(days=int(rng.integers(0, days))) for _ in range(max(user_days, 1))])

        for t in ts:
            pages = int(np.clip(rng.normal(5, 3), 1, 40))
            sess_len = float(np.clip(rng.normal(6, 4), 1, 90))  # minutes
            pay_fail = int(rng.random() < 0.03)
            support = int(rng.random() < 0.02)
            add_ons = int(rng.random() < 0.10)
            rows.append((cid, t.isoformat() + "Z", pages, sess_len, pay_fail, support, add_ons))

    df = pd.DataFrame(rows, columns=[
        "customer_id", "timestamp", "pages_viewed", "session_minutes", "payment_failure", "support_ticket", "add_on_click"
    ])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Synthesized telemetry sessions", extra={"rows": len(df), "path": out_path})
    return out_path


def aggregate_sessions_to_features(
    sessions_csv: str,
    out_path: str = "data/telemetry_features.csv"
) -> str:
    df = pd.read_csv(sessions_csv, parse_dates=["timestamp"])
    agg = df.groupby("customer_id").agg(
        sess_count=("timestamp", "count"),
        pages_mean=("pages_viewed", "mean"),
        minutes_sum=("session_minutes", "sum"),
        pay_fail_sum=("payment_failure", "sum"),
        support_sum=("support_ticket", "sum"),
        add_on_sum=("add_on_click", "sum"),
    ).reset_index()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    logger.info("Aggregated telemetry features", extra={"rows": len(agg), "path": out_path})
    return out_path
