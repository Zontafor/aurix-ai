from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


# Shared happy-path payload consistent with CustomerFeatures
BASE_PAYLOAD = {
    "customerID": "0001-AURIX",
    "tenure": 24,
    "MonthlyCharges": 75.5,
    "TotalCharges": 1812.0,
    "Contract": "One year",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Credit card (automatic)",
    "PaperlessBilling": "Yes",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
}


def test_health_ok():
    """Service liveness check."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") == "ok"
    assert "model_version" in body


def test_predict_risk_happy_path():
    """Happy-path test for /predict_risk."""
    response = client.post("/predict_risk", json=BASE_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["customer_id"] == BASE_PAYLOAD["customerID"]

    p = body["churn_probability"]
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0

# def test_recommend_action_happy_path():
#     """Happy-path test for /recommend_action."""
#     response = client.post("/recommend_action", json=BASE_PAYLOAD)
#     assert response.status_code == 200

#     body = response.json()
#     assert body["customer_id"] == BASE_PAYLOAD["customerID"]

#     p = body["churn_probability"]
#     assert 0.0 <= p <= 1.0

#     action = body["action"]
#     assert action in {"contact", "do_not_contact"}

#     profit = body["expected_profit"]
#     assert isinstance(profit, float)