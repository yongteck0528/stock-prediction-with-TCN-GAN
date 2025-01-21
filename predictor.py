import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_data, get_stock_data
import datetime
from preprocessing import y_scaler

def predict_stock_price(model, last_10_days, start_date):
    predictions, dates = [], []
    last_sequence = last_10_days.reshape(1, last_10_days.shape[0], last_10_days.shape[1])

    for _ in range(10):
        next_prediction = model(last_sequence)[0, 0]
        predictions.append(next_prediction)

        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_prediction

        next_date = pd.to_datetime(start_date) + pd.Timedelta(days=1)
        while next_date.weekday() >= 5: 
            next_date += pd.Timedelta(days=1)

        dates.append(next_date)
        start_date = next_date

    return predictions, dates

def predict_single_day_price(model, last_10_days):

    last_sequence = last_10_days.reshape(1, last_10_days.shape[0], last_10_days.shape[1])
    next_prediction = model(last_sequence)[0, 0]

    return next_prediction


def predict_single_date(model, predictions_df):
    st.title("Single Date Stock Price Prediction")
    input_date = st.date_input("Select a date for prediction:", value=datetime.date.today())
    input_date = pd.to_datetime(input_date)

    if input_date.weekday() >= 5:
        st.write("No transaction for weekends. Please select a weekday.")
        return

    today = datetime.date.today()

    business_days = pd.date_range(start=today, periods=30, freq='B') 
    if input_date not in business_days and input_date.date() > today:
        st.write("Please select a date within the next 30 business days.")
        return
    
    col1, col2 = st.columns(2)

    if input_date in predictions_df['Date'].values:
        predicted_price = predictions_df.loc[predictions_df['Date'] == input_date, 'Predicted Stock Price'].values[0]
        col1.markdown("<h3 style='color: green;'>Predicted Price</h3>", unsafe_allow_html=True)
        col1.markdown(f"<h3 style='color: green;'>Rp. {predicted_price:.2f}</h3>", unsafe_allow_html=True)
        col2.markdown("<h3 style='color: green;'>Actual Price</h3>", unsafe_allow_html=True)
        col2.markdown("<h3 style='color: green;'>-</h3>", unsafe_allow_html=True)
    else:
        stock_data = get_stock_data("ITMG.JK")
        actual_prices = stock_data[stock_data['Date'] == input_date]
        
        if not actual_prices.empty:
            actual_price = actual_prices['Close'].values[0]
            col2.markdown("<h3 style='color: green;'>Actual Price</h3>", unsafe_allow_html=True)
            col2.markdown(f"<h3 style='color: green;'>Rp. {actual_price:.2f}</h3>", unsafe_allow_html=True)

        historical_data = stock_data[stock_data['Date'] < input_date]

        if len(historical_data) < 10:
            st.error("Need at least 10 days of data for prediction.")
            return

        last_10_days = preprocess_data(historical_data)[-1]
        predicted_price_scaled = predict_single_day_price(model, last_10_days)
        predicted_price = y_scaler.inverse_transform(np.array([[predicted_price_scaled]]))[0, 0]
        col1.markdown("<h3 style='color: green;'>Predicted Price</h3>", unsafe_allow_html=True)
        col1.markdown(f"<h3 style='color: green;'>Rp. {predicted_price:.2f}</h3>", unsafe_allow_html=True)
