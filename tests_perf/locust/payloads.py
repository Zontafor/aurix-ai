import random

def random_payload():
    tenure = random.randint(0, 72)
    monthly = round(random.uniform(20, 120), 2)
    total = round(monthly * max(tenure, 1) * random.uniform(0.6, 1.2), 2)

    return {
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract_type": random.choice(["Month-to-month", "One year", "Two year"]),
        "internet_service": random.choice(["DSL", "Fiber optic", "None"]),
        "paperless_billing": random.choice([True, False]),
    }