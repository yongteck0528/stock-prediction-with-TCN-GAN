import pandas as pd
import os
from utils import preprocess_data
from predictor import predict_stock_price
import datetime
from preprocessing import x_scaler, y_scaler
import numpy as np

def store_10_day_predictions(model, stock_data):
    last_10_days = preprocess_data(stock_data)[-1]
    predictions, dates = predict_stock_price(model, last_10_days, datetime.date.today())
    future_predictions_scaled_back = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    predictions_df = pd.DataFrame({
        'Date': dates,
        'Predicted Stock Price': future_predictions_scaled_back.flatten()
    })

    csv_path = "dataset/predicted_10_days.csv"
    predictions_df.to_csv(csv_path, index=False)
    return predictions_df

def load_30_day_predictions():
    csv_path = "dataset/predicted_10_days.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=['Date'])
    return None
