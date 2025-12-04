# # src/api/recommend.py
# # AURIX-AI FastAPI router for action recommendations
# # based on churn risk and a simple profit model.

# from typing import Literal

# from fastapi import APIRouter

# # Reuse the same schema + prediction function as /predict_risk
# from .predict import CustomerFeatures, RiskResponse, predict_risk

# router = APIRouter(tags=["recommendation"])

# # --- Telco retention cost model (tunable constants) ---

# # Estimated present value of retaining a high-risk customer
# RETAINED_VALUE: float = 120.0

# # Cost of contacting / offering retention incentive
# CONTACT_COST: float = 20.0

# # Risk threshold chosen from the profit curve (Section VI)
# RISK_THRESHOLD: float = 0.35


# class RecommendationResponse(RiskResponse):
#     """
#     Extends the base RiskResponse with a recommended action and
#     expected profit under the Telco retention cost model.
#     """
#     decision: Literal["contact", "do_not_contact"]
#     expected_profit: float
#     threshold: float


# def _expected_profit(p_churn: float) -> float:
#     """
#     Simple profit model: treat churn probability as a proxy for uplift.
#     Expected profit of contacting is:
#         uplift * RETAINED_VALUE - CONTACT_COST
#     """
#     uplift = p_churn
#     return uplift * RETAINED_VALUE - CONTACT_COST


# @router.post("/recommend_action", response_model=RecommendationResponse)
# def recommend_action(customer: CustomerFeatures) -> RecommendationResponse:
#     """
#     Recommend whether to contact a customer and report the expected profit.

#     1. Call the existing /predict_risk logic to obtain p(churn | x).
#     2. Compute expected profit under the Telco retention model.
#     3. Recommend "contact" if risk is above RISK_THRESHOLD and
#        expected profit is positive; otherwise "do_not_contact".
#     """
#     # Step 1: reuse the existing risk prediction endpoint as a pure function
#     risk_resp: RiskResponse = predict_risk(customer)
#     p = risk_resp.churn_risk

#     # Step 2: profit calculation
#     profit = _expected_profit(p)

#     # Step 3: decision rule
#     decision: Literal["contact", "do_not_contact"]
#     if p >= RISK_THRESHOLD and profit > 0.0:
#         decision = "contact"
#     else:
#         decision = "do_not_contact"

#     return RecommendationResponse(
#         churn_risk=p,
#         model_version=risk_resp.model_version,
#         decision=decision,
#         expected_profit=profit,
#         threshold=RISK_THRESHOLD,
#     )

# src/api/recommend.py
# AURIX-AI: churn action recommendation endpoint.

from typing import Literal
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .predict import CustomerFeatures, RiskResponse, predict_risk

router = APIRouter(tags=["recommendation"])

# Simple, transparent profit model:
# - value_per_retained_customer: expected long-run margin if we successfully retain
# - cost_per_contact: cost of calling / reaching out to a customer
VALUE_PER_RETAINED_CUSTOMER = 100.0
COST_PER_CONTACT = 5.0


class RecommendResponse(BaseModel):
    """Response schema for /recommend_action."""
    customer_id: str
    churn_probability: float
    recommend_contact: bool
    decision_rule: str
    expected_profit_if_contact: float
    expected_profit_if_no_contact: float
    model_version: str


@router.post(
    "/recommend_action",
    response_model=RecommendResponse,
    summary="Recommend whether to contact a customer under the churn model.",
)
def recommend_action(features: CustomerFeatures) -> RecommendResponse:
    """
    Recommend a retention action for a single Telco customer.

    This endpoint reuses the /predict_risk logic to obtain the churn
    probability, then applies a simple threshold-based policy:

        - If p_churn is high enough that the expected gain from a
          successful retention exceeds the contact cost, we recommend
          "contact".
        - Otherwise, we recommend "do not contact".

    The per-customer profit model is intentionally simple but fully
    documented to satisfy the HW3 proof-of-service requirement.
    """
    # First, obtain the churn probability from the model.
    try:
        risk_resp: RiskResponse = predict_risk(features)  # type: ignore[misc]
    except HTTPException:
        # Propagate HTTP exceptions from predict_risk unchanged
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Failed to score customer: {exc}",
        ) from exc

    p = risk_resp.churn_probability  # <-- correct field name
    customer_id = risk_resp.customer_id
    model_version = risk_resp.model_version

    # Expected profit if we contact:
    #   profit_contact = p * VALUE_PER_RETAINED_CUSTOMER - COST_PER_CONTACT
    profit_contact = p * VALUE_PER_RETAINED_CUSTOMER - COST_PER_CONTACT

    # Expected profit if we do nothing (no contact cost, but we also do
    # not attempt to retain; for simplicity we treat this as zero).
    profit_no_contact = 0.0

    recommend = profit_contact > profit_no_contact
    decision_rule = (
        "contact_if_p_churn * value - cost > 0"
    )

    return RecommendResponse(
        customer_id=customer_id,
        churn_probability=p,
        recommend_contact=recommend,
        decision_rule=decision_rule,
        expected_profit_if_contact=profit_contact,
        expected_profit_if_no_contact=profit_no_contact,
        model_version=model_version,
    )