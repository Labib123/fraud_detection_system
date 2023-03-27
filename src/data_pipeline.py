import pandas as pd
from feature_engineering import engineer_features
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(file_path):
    df = pd.read_csv(file_path)

    df = engineer_features(df)

    return df


def preprocess_data(df):
    # Drop irrelevant columns from the data
    df = df.drop(['transaction_id', 'timestamp', 'user_id', 'device_id'], axis=1)

    df = pd.get_dummies(df, columns=['merchant_category', 'merchant_country', 'merchant_name', 'transaction_type'])

    X = df.drop('fraud', axis=1)
    y = df['fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def apply_transformations(transactions_df):

    transactions_df['transaction_datetime'] = pd.to_datetime(transactions_df['transaction_datetime'])

    transactions_df['transaction_date'] = transactions_df['transaction_datetime'].dt.date
    transactions_df['transaction_time'] = transactions_df['transaction_datetime'].dt.time

    transactions_df['merchant_category'] = transactions_df['merchant_category'].astype('category')
    transactions_df['merchant_country'] = transactions_df['merchant_country'].astype('category')

    transactions_df['transaction_amount'] = np.log(transactions_df['transaction_amount'])

    transactions_df.rename(columns={'is_fraudulent': 'label'}, inplace=True)

    return transactions_df
