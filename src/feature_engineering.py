import pandas as pd
import numpy as np


def engineer_features(df):
    df = engineer_transaction_frequency(df)

    df = engineer_transaction_amount_deviation(df)

    df = engineer_merchant_category_frequency(df)

    df = engineer_merchant_name_frequency(df)

    return df


def engineer_transaction_frequency(df):
    df['time_since_last_transaction'] = df.groupby('customer_id')['transaction_datetime'].diff().dt.total_seconds()

    window_size = 3  # the window size is set to 3 for demonstration purposes
    df['transaction_frequency'] = df.groupby('customer_id')['time_since_last_transaction'].rolling(window_size).apply(
        lambda x: np.mean(x), raw=False).fillna(method='bfill').values

    df = df.drop(['time_since_last_transaction'], axis=1)

    return df


def engineer_transaction_amount_deviation(df):
    mean_amounts = df.groupby('customer_id')['transaction_amount'].mean()

    std_amounts = df.groupby('customer_id')['transaction_amount'].std()

    transaction_amount_deviation = (df['transaction_amount'] - df['customer_id'].map(mean_amounts)) / df[
        'customer_id'].map(std_amounts)

    df['transaction_amount_deviation'] = transaction_amount_deviation

    return df


def engineer_merchant_category_frequency(df):
    merchant_category_frequency = df.groupby(['customer_id', 'merchant_category'])['merchant_category'].count() / \
                                  df.groupby('customer_id')['transaction_id'].count()

    merchant_category_frequency = merchant_category_frequency.reset_index()

    merchant_category_frequency.columns = ['merchant_category_' + str(col) for col in
                                           merchant_category_frequency.columns]

    df = pd.merge(df, merchant_category_frequency, on='customer_id')

    return df


def engineer_merchant_name_frequency(df):
    merchant_name_frequency = df.groupby(['customer_id', 'merchant_name'])['merchant_name'].count() / \
                              df.groupby('customer_id')['transaction_id'].count()

    merchant_name_frequency = merchant_name_frequency.reset_index()
    merchant_name_frequency = merchant_name_frequency.pivot(index='customer_id', columns='merchant_name',
                                                            values='merchant_name_frequency').fillna(0)

    merchant_name_frequency.columns = ['merchant_name_' + str(col) for col in merchant_name_frequency.columns]

    df = pd.merge(df, merchant_name_frequency, on='customer_id')

    return df
