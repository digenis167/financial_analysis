import streamlit as st
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

# Query Stock Data with given Ticker and Date range
def get_stock_data(ticker, start_date = '2024-07-22', end_date = '2024-08-22'):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

st.set_page_config(layout='wide')

# WebApp Title
st.title('Interactive Financial Stock Market Comperative Analysis Tool')

# Set Default Date Range and Display date_input
today = datetime.date.today()
last_year = today - relativedelta(years=5)

# Setup User input text boxes with defaults as AAPL and GOOGL
st.sidebar.header('User Input Options')
selected_stock = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()
selected_stock2 = st.sidebar.text_input('Enter Stock Ticker 2', 'GOOGL').upper()

# Query Stock Data
stock_data = get_stock_data(selected_stock, last_year, today) #, selected_date[0], selected_date[1])
stock_data2 = get_stock_data(selected_stock2, last_year, today) #, selected_date[0], selected_date[1])

# Create 2 columns for displaying the data
col1, col2 = st.columns(2)

# Display Stock Data
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

# OpenAI Comparative Performance query
if st.button('Comparative Performance'):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial assistant that will retrieve two tables of financial market data and will summarize the comparative performance in text, in full detail with highlights for each stock and also a conclusion with a markdown output. BE VERY STRICT ON YOUR OUTPUT"},
            {"role": "user", "content": f"This is the {selected_stock} stock data : {stock_data}, this is {selected_stock2} stock data: {stock_data2}"}
        ]
    )
    st.write(response.choices[0].message.content)