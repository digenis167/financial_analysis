import streamlit as st
import yfinance as yf
import datetime
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

# Tickers to use for the application
tickers_list = {'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Tesla': 'TSLA', 'NVIDIA Corp': 'NVDA', 'Intel Corp': 'INTC', 'Amazon.com Inc': 'AMZN',
                'Advanced Micro Devices, Inc.': 'AMD', 'Alphabet Inc Class C': 'GOOG', 'Marvell Technology Inc': 'MRVL', 'Arm Holdings PLC': 'ARM',
                'Airbnb Inc': 'ABNB', 'Qualcomm Inc': 'QCOM', 'Atlassian Corp': 'TEAM', 'Texas Instruments Inc': 'TXN', 'NXP Semiconductors NV': 'NXPI'
}
# Set the layout of the web application to wide
st.set_page_config(layout='wide')
# WebApp Title
st.title('Interactive Financial Stock Market Comperative Analysis Tool')

# Set Default Date Range and Display date_input
end_date = datetime.date.today()
start_date = end_date - relativedelta(months=6)
# Setup User input text boxes with defaults as AAPL and GOOGL
st.sidebar.header('User Input Options')
selected_stock = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()
selected_stock2 = st.sidebar.text_input('Enter Stock Ticker 2', 'GOOGL').upper()
# Query Stock Data
stock_data = get_stock_data(selected_stock, start_date, end_date)
stock_data2 = get_stock_data(selected_stock2, start_date, end_date)
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


st.divider()
st.sidebar.divider()
# Comparison Chart Price from selected year to date
st.subheader('Stock Price Chart')
st.sidebar.subheader('Stock Price Chart')
comp_chart_options = st.sidebar.multiselect(
    'Select Tickers',
    tickers_list.values(),
    ['AAPL', 'TSLA']
)

prev_years = st.sidebar.select_slider(
    'Select Years Range',
    options = np.arange(1,11)
)

# Create Figures to Compare Prices for selected range of years
fig = go.Figure()
for ticker in comp_chart_options:
    ticker_data = get_stock_data(ticker, datetime.date.today() - relativedelta(years=prev_years), datetime.date.today())
    ticker_data = pd.DataFrame(ticker_data).reset_index()
    fig.add_trace(
        go.Scatter(
            x = ticker_data['Date'],
            y = ticker_data['Close'],
            mode='lines',
            name=f"{ticker}"
        )
    )
fig.update_layout(
    title=dict(text="Stock Price Chart", font=dict(size=18)),
    xaxis=dict(title="Time", titlefont=dict(size=14)),
    yaxis=dict(title="Price", titlefont=dict(size=14))
)
st.plotly_chart(fig)


st.divider()
tickers = yf.Tickers(list(tickers_list.values()))
df = pd.DataFrame(yf.download(list(tickers_list.values()), period="1mo")).reset_index()

# Filter Close Date
df_close = df[['Close', 'Date']]
# Set new columns to the dataframe
cols = df_close.columns.get_level_values(1).to_list()
cols[-1] = 'Date'
df_close.columns = cols
# Melt the dataframe with 3 columns of date ticker and close_price
df_close = df_close.melt(id_vars='Date', var_name='Ticker', value_name='Close_Price')
# Pivot the table for Ticker as column and Closed prices based on Date
df_close = df_close.pivot(index='Ticker', columns='Date', values='Close_Price').reset_index()
# Remove the columns name
df_close.columns.name = None
# Change the format of columns with Timestamp type to datetime.date type
df_close.columns = [col.date() if isinstance(col, datetime.datetime) else col for col in df_close.columns]

# Get Market Open Dates and Add Current Price, Volume, and Market Cap. Find second last open market date based on current price
market_dates = [col for col in df_close.columns if isinstance(col, datetime.date)]
df_close['Current_Price'] = [tickers.tickers[tck].info['currentPrice'] for tck in df_close['Ticker']]
df_close['Volume'] = [tickers.tickers[tck].info['volume'] for tck in df_close['Ticker']]
df_close['Market_Cap'] = [tickers.tickers[tck].info['marketCap'] / 1e9 for tck in df_close['Ticker']]
last_market_open = end_date - relativedelta(days=1)
idx = len(market_dates) - 1
while idx >= 0:
    if (df_close['Current_Price'] == round(df_close[market_dates[idx]], 2)).eq(False).any():
        last_market_open = market_dates[idx]
        break
    idx -= 1

# Calculate the Mean Price, Day Delta and Day Performance and Top/Least 5 Performers
df_close['Mean_Close'] = df_close.loc[:, market_dates].mean(axis=1)
df_close['Day_Delta'] = df_close['Current_Price'] - df_close[last_market_open]
df_close['Day_Performance'] = (df_close['Day_Delta'] / df_close['Current_Price']) * 100
top_performers = df_close.nlargest(5, 'Day_Performance')
least_performers = df_close.nsmallest(5, 'Day_Performance')

# Display top performers last close and percentage of change between last two closed prices
st.subheader('Top Performers')
cols = st.columns(5)
for i in range(5):
    ticker = top_performers.iloc[i]['Ticker']
    cur_price = top_performers.iloc[i]['Current_Price']
    day_performance = top_performers.iloc[i]['Day_Performance']
    cols[i].metric(ticker, f"{cur_price:.2f}", f"{day_performance:.2f}%")

st.subheader('Least Performers')
cols = st.columns(5)
for i in range(5):
    ticker = least_performers.iloc[i]['Ticker']
    cur_price = top_performers.iloc[i]['Current_Price']
    day_performance = least_performers.iloc[i]['Day_Performance']
    cols[i].metric(ticker, f"{cur_price:.2f}", f"{day_performance:.2f}%")

st.divider()

# Compare stocks based on volume, price and market cap
fig = go.Figure(
    go.Scatter(
        x=df_close['Volume'] / 1e6, 
        y=df_close['Current_Price'], 
        mode='markers',
        marker=dict(
            size= df_close['Market_Cap'],
            sizeref = 2. * df_close['Market_Cap'].max() / (40 ** 2),
            sizemode='area',
            showscale=True,
            color=df_close['Market_Cap'],
            colorscale='Tealgrn',
            colorbar=dict(
                tickfont=dict(size=16),
                title='Market Cap (Billion)',
                titlefont=dict(size=20)
            )
        ),
        customdata=df_close[['Ticker', 'Market_Cap']],
        hovertemplate = 'Stock Ticker: %{customdata[0]}<br>Current Price: $%{y}<br>Market Cap: $%{customdata[1]:.2f}B<br>Volume: $%{x:.2f}M<extra></extra>',
        # 'bottles_sold=%{y}<br>zip_code=%{customdata[0]}<br>item_number=%{customdata[1]}<extra></extra>',
    )
)
fig.update_layout(
    title=dict(text="Stocks Comparison Volume, Price and Market Cap", font=dict(size=24)),
    xaxis = dict(
        title="Volume in Millions",
        titlefont=dict(size=20),
        tickfont=dict(size=16),
    ),
    yaxis = dict(
        title="Price($)",
        titlefont=dict(size=20),
        tickfont=dict(size=16),
    ),
)
st.plotly_chart(fig)

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