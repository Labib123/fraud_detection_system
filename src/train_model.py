import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from feature_engineering import engineer_features
from model_evaluation import evaluate_model
from sklearn.model_selection import train_test_split
from data_pipeline import apply_transformations

transactions_df = pd.read_csv('../data/transactions.csv')

transactions_df = apply_transformations(transactions_df)

print(transactions_df.dtypes)

transactions_df = engineer_features(transactions_df)

X_train, X_test, y_train, y_test = train_test_split(
    transactions_df.drop('fraudulent', axis=1),
    transactions_df['fraudulent'],
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

evaluate_model(model, X_test, y_test)

joblib.dump(model, 'models/rf_model.pkl')
