from faker import Faker
import pandas as pd
import random

fake = Faker()

n_transactions = 10000

transaction_schema = {
    'transaction_id': [fake.uuid4() for _ in range(n_transactions)],
    'customer_id': [fake.uuid4() for _ in range(n_transactions)],
    'transaction_datetime': [fake.date_time_this_month() for _ in range(n_transactions)],
    'transaction_amount': [random.uniform(0, 10000) for _ in range(n_transactions)],
    'merchant_id': [fake.uuid4() for _ in range(n_transactions)],
    'merchant_category': [fake.job() for _ in range(n_transactions)],
    'merchant_country': [fake.country() for _ in range(n_transactions)],
    'merchant_name': [fake.company() for _ in range(n_transactions)],
    'is_fraudulent': [0 for _ in range(int(n_transactions * 0.95))] + [1 for _ in range(int(n_transactions * 0.05))]
}

transactions_df = pd.DataFrame(transaction_schema)

transactions_df.to_csv('transactions.csv', index=False)
