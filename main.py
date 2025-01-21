import streamlit as st
from load_model import load_tcn_model
from utils import get_stock_data
from predictor import predict_single_date
from storage import store_10_day_predictions
from show import show_documentation, show_stock_table, show_predict_table

def main():
    model = load_tcn_model()
    stock_data = get_stock_data("ITMG.JK")
    stock_data.to_csv('dataset/stock_data.csv', index=False)
    next_10_days_df = store_10_day_predictions(model, stock_data)

    st.sidebar.markdown("<h1 style='color:white;'>ITMG.JK Stock Prediction</h1>", unsafe_allow_html=True)
    page = st.sidebar.radio("Go to", ["Home", "1-Day Prediction", "10-Days Prediction", "Documentation"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("[GitHub Repository](https://github.com/yongteck0528)") 
    # st.sidebar.write("Try 28 October 2024 for prediction") 

    if page == "Home":
        show_stock_table()
    elif page == "1-Day Prediction":
        predict_single_date(model, next_10_days_df)
    elif page == "10-Days Prediction":
        show_predict_table(next_10_days_df) 
    elif page == "Documentation":
        show_documentation()
    

if __name__ == "__main__":
    main()
