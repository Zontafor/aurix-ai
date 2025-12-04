with src as (
    select * from {{ ref('telco_sample') }}
)

select
    -- identifiers
    customerID                          as customer_id,

    -- target label
    case
        when Churn = 'Yes' then 1
        when Churn = 'No'  then 0
        else null
    end                                 as churn,

    -- core numeric fields
    cast(tenure as integer)             as tenure_months,
    cast(MonthlyCharges as numeric)     as monthly_charges,
    cast(TotalCharges as numeric)       as total_charges,

    -- contract and billing
    Contract                            as contract,
    PaperlessBilling                    as paperless_billing,
    PaymentMethod                       as payment_method,

    -- internet service + add-ons
    InternetService                     as internet_service,
    OnlineSecurity                      as online_security,
    OnlineBackup                        as online_backup,
    DeviceProtection                    as device_protection,
    TechSupport                         as tech_support,
    StreamingTV                         as streaming_tv,
    StreamingMovies                     as streaming_movies,

    -- phone service
    PhoneService                        as phone_service,
    MultipleLines                       as multiple_lines,

    -- demographics
    gender,
    SeniorCitizen,
    Partner,
    Dependents

from src