import yfinance as yf
import pandas as pd
from preprocessing import x_scaler, sliding_window

def get_stock_data(ticker):
    stock_data = yf.download(ticker, interval='1d')
    stock_data.reset_index(inplace=True)
    return stock_data

def preprocess_data(stock_data, window_size=10):
    stock_data = stock_data[['Close', 'Open', 'High', 'Low']]
    stock_data_scaled = x_scaler.transform(stock_data.values)
    preprocessed_data, _, _ = sliding_window(stock_data_scaled, stock_data_scaled[:, 0], window_size)
    return preprocessed_data
