-- models/staging/stg_telco_customers.sql
with source as (
    select * from {{ ref('telco_sample') }}
),
cleaned as (
    select
        cast(customerID as text) as customer_id,
        try_cast(tenure as integer) as tenure,
        try_cast(MonthlyCharges as double) as monthly_charges,
        try_cast(TotalCharges as double) as total_charges,
        case when Churn in ('Yes', '1', 'true', 'TRUE') then 1 else 0 end as churn_flag,
        Contract as contract_type,
        PaymentMethod as payment_method,
        InternetService as internet_service
    from source
)
select * from cleaned;
