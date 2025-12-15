# src/models/profit_loss.py
# Profit objective helpers and ROI utilities shared by training and API layers.

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass
class ProfitConfig:
    margin_per_customer: float = 100.0
    contact_cost: float = 5.0
    max_contact_rate: float = 0.30
    uplift_effect_guess: float = 0.20


def profit_curve_from_scores(scores: np.ndarray, cfg: ProfitConfig) -> pd.DataFrame:
    order = np.argsort(-scores)
    s_sorted = scores[order]
    n = len(s_sorted)
    ks = np.arange(1, int(cfg.max_contact_rate * n) + 1)
    profits = []
    for k in ks:
        expected_retained = cfg.uplift_effect_guess * s_sorted[:k].sum()
        profit = cfg.margin_per_customer * expected_retained - cfg.contact_cost * k
        profits.append(profit)
    return pd.DataFrame({"k": ks, "contact_rate": ks / n, "profit": profits})


def argmax_profit(curve: pd.DataFrame) -> Tuple[int, float, float]:
    """Return (k_at_max, contact_rate_at_max, max_profit)."""
    idx = int(np.argmax(curve["profit"].values))
    return int(curve.loc[idx, "k"]), float(curve.loc[idx, "contact_rate"]), float(curve.loc[idx, "profit"])


def roi_at_k(k: int, scores: np.ndarray, cfg: ProfitConfig) -> float:
    """Simple ROI: (gain - cost) / cost for the top-k contacts."""
    order = np.argsort(-scores)
    s = scores[order][:k].sum()
    gain = cfg.margin_per_customer * cfg.uplift_effect_guess * s
    cost = cfg.contact_cost * k
    if cost == 0:
        return 0.0
    return (gain - cost) / cost
