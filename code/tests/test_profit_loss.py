# tests/test_profit_loss.py
import numpy as np
from src.models.profit_loss import ProfitConfig, profit_curve_from_scores, argmax_profit, roi_at_k

def test_profit_curve_and_roi():
    cfg = ProfitConfig(margin_per_customer=100.0, contact_cost=5.0, max_contact_rate=0.5, uplift_effect_guess=0.2)
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.05, 0.7, 0.6, 0.3, 0.4, 0.05])
    curve = profit_curve_from_scores(scores, cfg)
    assert not curve.empty
    k, rate, pmax = argmax_profit(curve)
    assert 1 <= k <= len(scores)
    roi = roi_at_k(k, scores, cfg)
    assert isinstance(roi, float)
