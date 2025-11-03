# aurix-ai
CMU Tepper Machine Learning for Business Applications Software Demo

## Repo
aurix-ai/
│
├── README.md                         ← Overview, demo GIF, quickstart
├── LICENSE                           ← MIT or CMU-style license
├── .gitignore                        ← Python, Docker, Jupyter, etc.
├── docker-compose.yml                ← Spins up API, DB, MLflow, Feast, Grafana
├── Dockerfile                        ← For FastAPI inference service
├── requirements.txt                  ← scikit-learn, xgboost, feast, mlflow, etc.
│
├── src/
│   ├── api/
│   │   ├── main.py                   ← FastAPI app w/ /train and /predict endpoints
│   │   └── routes/
│   │       ├── train.py
│   │       └── predict.py
│   ├── models/
│   │   ├── churn_model.py            ← Training scripts (XGBoost, logistic)
│   │   ├── uplift_model.py           ← Uplift modeling (CausalML / EconML)
│   │   └── profit_loss.py            ← Profit-max objective
│   ├── features/
│   │   ├── feature_builder.py        ← Feature engineering (Feast ingest)
│   │   ├── schema.yaml               ← Feature definitions
│   │   └── transformations.py
│   ├── data/
│   │   ├── ingest_telco.py           ← Loads IBM Telco dataset or webhooks
│   │   ├── telemetry_synth.py        ← Synthetic event generator
│   │   └── sql/
│   │       ├── create_tables.sql
│   │       └── sample_queries.sql
│   ├── monitoring/
│   │   ├── drift_check.py            ← EvidentlyAI drift monitoring
│   │   └── profit_dashboard.py       ← Streamlit dashboard
│   └── utils/
│       ├── encryption.py             ← AES-GCM, Vault/Doppler integration
│       └── logger.py                 ← Structured logging
│
├── notebooks/
│   ├── 01_eda_telco.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_profit_uplift.ipynb
│   └── 04_explainability.ipynb
│
├── dbt_project/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   └── marts/
│   └── seeds/
│       └── telco_sample.csv
│
└── docs/
    ├── architecture_diagram.svg
    ├── data_flow.pdf
    └── proposal_latex/
        └── aurix_ai_proposal.tex     ← IEEE double-column version


## Pipeline
            ┌──────────────┐     Batch (Airbyte/cron)      ┌──────────────┐
 SaaS/CSV → │  Ingestion   ├──────────────────────────────→ │   Raw Zone   │
 Webhooks → │  (FastAPI)   │  Webhooks (Stripe/Zendesk)     │  DO Spaces   │
  Segment → │ + HMAC auth  └──────────────────────────────→ │   (S3)      │
            └──────┬───────┘                                └──────┬───────┘
                   │                                                  │
                   │                 dbt transforms                   │
                   │                                                  ▼
               ┌────▼─────┐      ┌───────────────────┐        ┌──────────────┐
               │ Staging  │◄─────┤  Orchestrator     ├──────► │  Warehouse   │
               │ Postgres │      │  (cron/Temporal)  │        │ Postgres/CH  │
               └────┬─────┘      └───────────────────┘        └────┬─────────┘
                    │                         Feature views          │
                    │                                              ┌─▼─────────┐
                    │                    ┌───────────────────┐     │ Feature   │
                    └──────────────────► │  Feature Store    │ ◄──►│ Store     │
                                         │  (Feast on PG)    │     │ (PG/Redis)│
                                         └────────┬──────────┘     └─┬─────────┘
                                                  │                  │
                                       Offline training          Online lookup
                                                  │                  │
                                          ┌───────▼────────┐        │
                                          │  Training Job  │        │
                                          │ (sklearn/XGB)  │        │
                                          └───────┬────────┘        │
                                      Models + metrics to            │
                                      MLflow + Spaces (S3)           │
                                                  │                  │
                                           ┌──────▼──────┐     ┌────▼───────┐
                                           │  Model Svc  │◄────┤  API/Edge  │
                                           │ (FastAPI)   │     │ NGINX+Caddy│
                                           └──────┬──────┘     └────┬───────┘
                                    SHAP, profit/uplift            │
                                    recommendations →               │
                                    Dashboards (Streamlit)          │
                                                                   ▼
                                                         BI (Metabase/Looker)
## Local Quickstart

```bash
# Clone and install
git clone https://github.com/your-org/aurix-ai.git
cd aurix-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Prepare data
python aurix_data_cleaner.py
python aurix_feature_engineering.py

# Train models
python aurix_model_train_xgb.py        # Optimized XGBoost + uplift
python aurix_evaluate.py               # Aggregate metrics and plots

# Run the API service
docker-compose up --build
```

## tbc...
