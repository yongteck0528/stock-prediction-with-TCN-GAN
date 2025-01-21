import streamlit as st
import base64
import matplotlib.pyplot as plt
from utils import get_stock_data
import json

def show_documentation():
    st.title("Documentation")

    with open("asset/metrics/model_evaluation_C2.json", "r") as f:
        metrics = json.load(f)

    col1, col2 = st.columns(2)

    with col1:
        st.image("asset/img/A4_B3_C2_train_plot.png", caption="Predict_train", use_column_width=True)
        st.subheader('Train Dataset Metrics')
        st.markdown(f"**RMSE**: {metrics['RMSE_train']:.2f}\n")
        st.markdown(f"**MAPE**: {metrics['MAPE_train']:.2f}%\n")
    with col2:    
        st.image("asset/img/A4_B3_C2_test_plot.png", caption="Predict_test", use_column_width=True)
        st.subheader('Test Dataset Metrics')
        st.markdown(f"**RMSE**: {metrics['RMSE_test']:.2f}\n")
        st.markdown(f"**MAPE**: {metrics['MAPE_test']:.2f}%\n")

    st.markdown('---')
    st.subheader('Generator and Discriminator Loss')
    st.image("asset/img/training_loss_plot_C2.png", caption='Training Loss', use_column_width=True)

def show_stock_table():
    st.title("ITMG.JK Stock Prices")
    stock_data = get_stock_data("ITMG.JK")
    stock_data = stock_data.iloc[::-1].reset_index(drop=True)

    first_date = stock_data['Date'].iloc[-1]
    st.markdown(f"Data since {first_date}")
    st.dataframe(stock_data, width=800, height=400)

    st.markdown('---')
    st.subheader('ITMG Close Price Trend')
    plt.figure(figsize=(10, 5))  
    plt.plot(stock_data['Date'], stock_data['Close'], color='blue', marker='', label='Close Price')
    plt.title("Closing Prices of ITMG.JK")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)
    plt.close()

def show_predict_table(predictions_df):
    st.title("Stock Price Prediction")
    st.write("Predictions for the next 10 days")

    if predictions_df is not None:
        st.dataframe(predictions_df, width=800, height=300)

        st.markdown('---')

        st.subheader('Prediction Trend')
        plt.figure(figsize=(10, 5))
        plt.plot(predictions_df['Date'], predictions_df['Predicted Stock Price'], label='Predicted Price')
        plt.title("Predicted Stock Prices for the Next 10 Days")
        plt.xlabel("Date")
        plt.ylabel("Predicted Stock Price")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        plt.close()
    else:
        st.error("Prediction data is not available.")