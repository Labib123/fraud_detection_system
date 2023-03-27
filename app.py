from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.feature_engineering import engineer_features

app = Flask(__name__)

model = joblib.load('models/rf_model.pkl')


@app.route('/', methods=['POST'])
def predict():
    transaction_data = request.json['transaction_data']

    transaction_df = pd.DataFrame(transaction_data)

    transaction_df = engineer_features(transaction_df)

    fraud_predictions = model.predict(transaction_df)

    fraud_predictions = fraud_predictions.tolist()

    return jsonify({'fraud_predictions': fraud_predictions})
