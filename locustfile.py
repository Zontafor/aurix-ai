# locustfile.py
# Locust load test for the AURIX-AI churn service.
# locust -f locustfile.py --host=http://localhost:8080
# Terminal testing:
# locust -f locustfile.py \
#   --headless \
#   -u 2000 \
#   -r 8 \
#   -t 3h \
#   --host=http://localhost:8080 \
#   --loglevel=INFO

from locust import HttpUser, task, between

BASE_PAYLOAD = {
    "customerID": "locust-user",
    "tenure": 18,
    "MonthlyCharges": 65.0,
    "TotalCharges": 1170.0,
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


class ChurnApiUser(HttpUser):
    """
    Synthetic user that repeatedly calls the churn API.

    The host is set when launching Locust, e.g.
        locust -f locustfile.py --host=http://localhost:8080
    """

    wait_time = between(0.1, 0.5)

    @task(3)
    def predict_risk(self):
        self.client.post("/predict_risk", json=BASE_PAYLOAD)

    @task(1)
    def recommend_action(self):
        self.client.post("/recommend_action", json=BASE_PAYLOAD)