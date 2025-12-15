def test_predict_risk_happy_path(client, base_url):
    # Replace fields with whatever your API expects
    payload = {
        "customerID": "pytest-user",
        "tenure": 12,
        "MonthlyCharges": 75.3,
        "TotalCharges": 900.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
    }

    r = client.post(f"{base_url}/predict_risk", json=payload, timeout=10)

    assert r.status_code == 200, r.text
    data = r.json()

    # Minimal “happy path” validation
    assert "churn_probability" in data, data
    p = data["churn_probability"]
    assert isinstance(p, (int, float)), type(p)
    assert 0.0 <= p <= 1.0

    assert "customer_id" in data, data
    assert isinstance(data["customer_id"], str)

    assert "model_version" in data, data
    assert isinstance(data["model_version"], str)

def test_predict_risk_missing_field_returns_4xx(client, base_url):
    payload = {"tenure_months": 12}  # intentionally incomplete
    r = client.post(f"{base_url}/predict_risk", json=payload, timeout=10)
    assert r.status_code in (400, 422), r.text

def test_predict_risk_bad_type_returns_4xx(client, base_url):
    payload = {
        "tenure_months": "twelve",  # wrong type
        "monthly_charges": 75.3,
        "total_charges": 900.0,
    }
    r = client.post(f"{base_url}/predict_risk", json=payload, timeout=10)
    assert r.status_code in (400, 422), r.text