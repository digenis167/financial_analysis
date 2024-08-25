import streamlit as st
import yfinance as yf
# import datetime
from dateutil.relativedelta import relativedelta
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
import numpy as np

client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

# Query Stock Data with given Ticker and Date range
def get_stock_data(ticker, start_date = '2024-07-22', end_date = '2024-08-22'):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

tickers_list = {'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Tesla': 'TSLA', 'NVIDIA Corp': 'NVDA', 'Intel Corp': 'INTC', 'Amazon.com Inc': 'AMZN',
                'Advanced Micro Devices, Inc.': 'AMD', 'Alphabet Inc Class C': 'GOOG', 'Marvell Technology Inc': 'MRVL', 'Arm Holdings PLC': 'ARM',
                'Airbnb Inc': 'ABNB', 'Qualcomm Inc': 'QCOM', 'Atlassian Corp': 'TEAM', 'Texas Instruments Inc': 'TXN', 'NXP Semiconductors NV': 'NXPI'
}


st.set_page_config(layout='wide')

# WebApp Title
st.title('Interactive Financial Stock Market Comperative Analysis Tool')

# Set Default Date Range and Display date_input
# end_date = datetime.date.today()
# start_date = end_date - relativedelta(months=6)

# Setup User input text boxes with defaults as AAPL and GOOGL
st.sidebar.header('User Input Options')
selected_stock = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()
selected_stock2 = st.sidebar.text_input('Enter Stock Ticker 2', 'GOOGL').upper()

# Query Stock Data
stock_data = get_stock_data(selected_stock)
stock_data2 = get_stock_data(selected_stock2)

# Create 2 columns for displaying the data
col1, col2 = st.columns(2)

# Display Selected Stock Data
with col1:
    st.subheader(f"Displaying data for: {selected_stock}")
    st.dataframe(stock_data, use_container_width=True)
    chart_type = st.sidebar.selectbox(f"Select chart type for {selected_stock}", ["Line", "Bar"])
    if chart_type == 'Line':
        st.line_chart(stock_data['Close'])
    elif chart_type == 'Bar':
        st.bar_chart(stock_data['Close'])
with col2:
    st.subheader(f"Displaying data for: {selected_stock2}")
    st.dataframe(stock_data2, use_container_width=True)
    chart_type = st.sidebar.selectbox(f"Select chart type for {selected_stock2}", ["Line", "Bar"])
    if chart_type == 'Line':
        st.line_chart(stock_data2['Close'])
    elif chart_type == 'Bar':
        st.bar_chart(stock_data2['Close'])