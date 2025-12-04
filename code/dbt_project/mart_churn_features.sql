-- models/marts/mart_churn_features.sql
with base as (
    select * from {{ ref('stg_telco_customers') }}
),
features as (
    select
        customer_id,
        tenure,
        monthly_charges,
        total_charges,
        churn_flag,
        case when contract_type ilike '%Two year%' then 2
             when contract_type ilike '%One year%' then 1
             else 0 end as contract_type_code,
        case when internet_service ilike '%fiber%' then 2
             when internet_service ilike '%dsl%' then 1
             else 0 end as internet_service_code
    from base
)
select * from features;
