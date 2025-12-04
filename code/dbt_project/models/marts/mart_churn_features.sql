with base as (
    select *
    from {{ ref('stg_telco_customers') }}
),

engineered as (
    select
        customer_id,
        churn,

        -- numeric features
        tenure_months,
        monthly_charges,
        total_charges,
        case
            when tenure_months > 0
            then total_charges / nullif(tenure_months, 0)
            else null
        end as avg_monthly_spend,

        -- contract structure
        case when contract in ('One year', 'Two year') then 1 else 0 end as is_long_contract,

        -- billing & payment
        case when lower(payment_method) like '%electronic check%' then 1 else 0 end as is_electronic_check,
        case when paperless_billing = 'Yes' then 1 else 0 end as is_paperless_billing,

        -- internet & add-ons
        case when internet_service = 'Fiber optic' then 1 else 0 end as has_fiber_internet,
        case when online_security = 'Yes' then 1 else 0 end as has_online_security,
        case when tech_support = 'Yes' then 1 else 0 end as has_tech_support,

        -- simple streaming indicators
        case when streaming_tv = 'Yes' then 1 else 0 end as has_streaming_tv,
        case when streaming_movies = 'Yes' then 1 else 0 end as has_streaming_movies

    from base
)

select * from engineered