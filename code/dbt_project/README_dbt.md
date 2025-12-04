# dbt Integration (AURIX-AI)

This project ships with a minimal dbt structure suitable for local development. The default recommendation is to use DuckDB for quick testing; you can switch to Postgres by creating a dbt profile.

# aurix_churn_dbt

dbt project for the Telco churn case study in *Machine Learning for Business Applications*.

The goal of this project is to demonstrate a minimal, production-style data
transformation pipeline that:

1. Ingests a raw Telco churn CSV extract (`telco_sample.csv`) as a dbt seed.
2. Builds a cleaned, type-safe staging view (`stg_telco_customers`).
3. Produces a feature-ready mart table (`mart_churn_features`) that can be
   consumed by the XGBoost / uplift modeling API in `src/api/main.py`.

## Layout

- `dbt_project.yml` – main dbt project configuration.
- `seeds/telco_sample.csv` – raw Telco churn sample (copied from the course repo).
- `models/staging/stg_telco_customers.sql` – staging model that cleans and
  normalizes the raw CSV.
- `models/marts/mart_churn_features.sql` – mart model with engineered features
  used by the churn propensity and uplift models.
- `models/schema.yml` – documentation + tests for both models.

## Prerequisites

- Python environment with `dbt-core` and the appropriate adapter installed
  (e.g., `dbt-postgres`, `dbt-bigquery`, or `dbt-snowflake`).
- A configured `profiles.yml` entry named `aurix_churn` that points to your
  target data warehouse.

## Quick start

From the `dbt_project` directory:

```bash
# 1. Check connection
dbt debug

# 2. Load the Telco seed into the warehouse
dbt seed --select telco_sample

# 3. Build staging + mart models
dbt run

# 4. Run basic tests
dbt test
```

## Directory
- `dbt_project.yml`
- `models/staging/stg_telco_customers.sql`
- `models/marts/mart_churn_features.sql`
- `seeds/telco_sample.csv`

## DuckDB (local) profile
Create `~/.dbt/profiles.yml`:

```yaml
aurix_ai:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ./aurix_ai.duckdb
```

Then run:
```bash
cd dbt_project
dbt seed
dbt run
```

## Postgres profile (optional)
```yaml
aurix_ai:
  target: pg
  outputs:
    pg:
      type: postgres
      host: localhost
      user: aurix
      password: aurix
      port: 5432
      dbname: aurix
      schema: public
```
Then:
```bash
dbt seed --profiles-dir ~/.dbt
dbt run --profiles-dir ~/.dbt
```

The staging model cleans common Telco fields and the mart model aggregates features per customer.
