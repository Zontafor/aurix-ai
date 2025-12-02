# AURIX-AI API Usage Examples

## Train
Retrain the model using data mounted at `./data`.
If `data/Telco_customer_churn.xlsx` exists, it will be cleaned; otherwise `data/Telco_clean.csv` is used.

```bash
curl -X POST http://localhost:8080/train   -H "Content-Type: application/json"
```

Expected response:
```json
{
  "status": "trained",
  "artifacts": [
    "aurix_churn_model_xgb.joblib",
    "aurix_uplift_model_treated_xgb.joblib",
    "aurix_uplift_model_control_xgb.joblib",
    "aurix_churn_eval_xgb.json",
    "data/aurix_features.csv"
  ],
  "metrics_path": "aurix_churn_eval_xgb.json"
}
```

## Predict
Send records as JSON. The service aligns columns to the trained feature set.

```bash
curl -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -d '{
    "records": [
      {"tenure": 12, "monthlycharges": 70.5, "totalcharges": 845.3, "contract_type": 1},
      {"tenure": 3,  "monthlycharges": 95.0, "totalcharges": 300.0, "contract_type": 0}
    ]
  }'
```

Example response:
```json
{
  "n": 2,
  "churn_prob": [0.182, 0.621],
  "uplift": [0.041, -0.013]
}
```
