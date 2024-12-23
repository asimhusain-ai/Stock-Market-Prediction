# Import Libraries
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model('stock_data.keras')

# Set up Streamlit page configuration
st.set_page_config(
    page_title='Stock Price Prediction',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        background: linear-gradient(135deg, #71b7e6, #9b59b6);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
    }
    .stButton>button {
        background-color: #009688;
        border: none;
        border-radius: 10px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00796b;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .button-container {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-top: 20px;
    }
    .crypto-link-box {
        width: 70%;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #BCC6CC;
        padding: 5px;
        border-radius: 10px;
        margin-top: 35px;
        text-align: center;
        border: 0px solid #ccc;
        transition: all 0.3s ease-in-out;
    }
    .crypto-link-box:hover {
        background-color: #f0f0f0;
        animation: bounce 0.5s ease-in-out infinite alternate;
    }
    @keyframes bounce {
        0% {
            transform: translateY(0);
        }
        100% {
            transform: translateY(-10px);
        }
    }
    .crypto-link {
        color: #000000;
        font-weight: bold;
        text-decoration: none !important;
        font-size: 0.9em;
        transition: color 0.3s ease;
    }
    .crypto-link:hover {
        color: #333333;
    }
    a.crypto-link-box, a.crypto-link-box:hover {
        text-decoration: none;
    }
    .crypto-link-box::after {
        content: '→';
        margin-left: 10px;
        font-size: 1.2em;
        color: #000000; /* Set arrow color to white */
        transition: transform 0.3s ease-in-out, opacity 0.3s ease-in-out;
        opacity: 0;
        transform: translateX(-10px);
    }
    .crypto-link-box:hover::after {
        opacity: 1;
        transform: translateX(0);
    }

    /* Admin button styles (circular design with smaller logo) */
    .sidebar-button-container {
        position: absolute;
        bottom: -430px; /* Adjust to place at the bottom */
        width: 20%;
        text-align: center;
    }
    .sidebar-button {
        display: inline-block;
        padding: 15px 15px;
        font-size: 20px;
        font-weight: bold;
        background-color: #ffffff;
        color: white;
        border-radius: 50%; /* Circular shape */
        transition: all 0.3s ease-in-out;
        text-decoration: none; /* Remove underline */
        width: 35px; /* Size of the button */
        height: 35px; /* Size of the button */
        line-height: 30px; /* Vertical centering */
        text-align: center; /* Horizontal centering */
        background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJ0hJNr9dy5PpovgYX-9omHtQdDzeSM8huHXz-nOby2tx8r7ighMv23RoPIHBE9GLG1ZE&usqp=CAU');
        background-size: 60%; /* Smaller logo size */
        background-position: center; /* Center the logo */
        background-repeat: no-repeat; /* Prevent image repetition */
    }
    .sidebar-button:hover {
        background-color: #f0f0f0;
        color: black;
        transform: scale(1.1); /* Increase size on hover */
    }
    # .sidebar-button::after {
    #     content: '→';
    #     margin-left: 10px;
    #     font-size: 1.2em;
    #     color: white; /* Set arrow color to white */
    #     transition: transform 0.3s ease-in-out, opacity 0.3s ease-in-out;
    #     opacity: 0;
    #     transform: translateX(-10px);
    # }
    .sidebar-button:hover::after {
        opacity: 1;
        transform: translateX(0);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header('Stock Price Prediction')
stock_symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 'BRK-A', 'BRK-B', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT', 'PG', 'MA', 'DIS', 'HD', 'PYPL', 'VZ', 'ADBE', 'NFLX', 'PFE', 'CMCSA', 'PEP', 'INTC', 'KO', 'CSCO', 'T', 'MRK', 'ABT', 'CVX', 'XOM', 'NKE', 'LLY', 'MCD', 'MDT', 'TXN', 'AVGO', 'CRM', 'HON', 'UNH', 'TMO', 'AMGN', 'NEE', 'DHR', 'QCOM', 'LIN', 'ACN', 'SPGI', 'ORCL', 'WBA', 'IBM', 'ISRG', 'CAT', 'DE', 'LOW', 'NOW', 'LMT', 'PM', 'MMM', 'FISV', 'BLK', 'COST', 'UPS', 'RTX', 'MO', 'BKNG', 'CCI', 'USB', 'MS', 'BA', 'GS', 'PLD', 'SCHW', 'ZTS', 'CB', 'GILD', 'CL', 'PNC', 'BMY', 'MU', 'CCI', 'GPN', 'BDX', 'CI', 'APD', 'DUK', 'C', 'PSX', 'TGT', 'GE', 'SO', 'EXC', 'FDX', 'MET', 'COP']
stock = st.sidebar.selectbox('Select | Search Stock Symbol', stock_symbols)
start = st.sidebar.text_input('Start Date', '2022-01-01')
end = st.sidebar.text_input('End Date', '2024-01-01')

# Buttons container
st.sidebar.markdown("""
<div class="button-container">
    <a class="crypto-link-box" href="https://finance.yahoo.com/?guccounter=1">
        <span class="crypto-link">🇸 🇾 🇲 🇧 🇴 🇱 🇸 </span>
    </a>
    <a class="crypto-link-box" href="https://coinmarketcap.com/converter/">
        <span class="crypto-link">🇨 🇦 🇱 🇨 🇺 🇱 🇦 🇹 🇴 🇷 </span>
    </a>
</div>

 <div class="sidebar-button-container">
        <a href="http://asimhusain.zya.me" class="sidebar-button" target="_blank"></a>
    </div>

""", unsafe_allow_html=True)

# Fetch stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data (USD)')
st.dataframe(data.style.set_table_styles([
    {'selector': 'th:nth-child(5)', 'props': [('min-width', '200px')]},
    {'selector': 'th:nth-child(6)', 'props': [('min-width', '200px')]},
    {'selector': 'th:nth-child(2)', 'props': [('min-width', '200px')]},
    {'selector': 'th:nth-child(4)', 'props': [('min-width', '200px')]}
]),
      height=500,
    width=900        
             )

# Data preparation
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# Moving averages and plots
st.markdown('<hr>', unsafe_allow_html=True)
st.subheader('Stock Price |vs| 50 Days Moving Average')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(30, 15))
plt.plot(ma_50_days, 'r', label='Moving Average')
plt.plot(data.Close, 'g', label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Stock Price |vs| 50 Days Moving Average')
plt.grid(True) 
plt.legend()
st.pyplot(fig1)


st.markdown('<hr>', unsafe_allow_html=True)
st.subheader('Stock Price |vs| 50 Days Moving Average |vs| 100 Days Moving Average')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(30, 15))
plt.plot(data.index, ma_50_days, 'r', label='50-Day MA')
plt.plot(data.index, ma_100_days, 'b', label='100-Day MA')
plt.plot(data.index, data.Close, 'g', label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Stock Price |vs| 50 Days Moving Average |vs| 100 Days Moving Average')
plt.grid(True)
plt.legend()
st.pyplot(fig2)

st.markdown('<hr>', unsafe_allow_html=True)
st.subheader('Stock Price |vs| 100 Days Moving Average |vs| 200 Days Moving Average')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(30, 15))
plt.plot(data.index, ma_100_days, 'r', label='100-Day MA')
plt.plot(data.index, ma_200_days, 'b', label='200-Day MA')
plt.plot(data.index, data.Close, 'g', label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Stock Price |vs| 100 Days Moving Average |vs| 200 Days Moving Average')
plt.grid(True)
plt.legend()
st.pyplot(fig3)


# Prepare data for prediction
x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Prediction
predict = model.predict(x)
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Plot prediction vs original
st.markdown('<hr>', unsafe_allow_html=True)
st.subheader('Original Stock Price |vs| Predicted Stock Price')
fig4 = plt.figure(figsize=(30, 15))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.title('Original Stock Price |vs| Predicted Stock Price')
plt.grid(True)
plt.legend()
st.pyplot(fig4)



st.markdown('<hr>', unsafe_allow_html=True)

# Example data (replace this with your actual stock data)
date_range = pd.date_range(start=start, end=end, freq='B')  # 'B' for business days
data = pd.DataFrame({
    'Open': np.random.uniform(100, 500, len(date_range)),
    'Close': np.random.uniform(100, 500, len(date_range))
}, index=date_range)

# Create a complete date range for business days
full_dates = pd.date_range(start=start, end=end, freq='B')
full_data = pd.DataFrame(index=full_dates, columns=['Original Price', 'Predicted Price'])

# Fill the DataFrame
full_data['Original Price'] = data['Open'].reindex(full_dates)
full_data['Predicted Price'] = data['Close'].reindex(full_dates)
full_data['Difference'] = abs(full_data['Original Price'] - full_data['Predicted Price'])

# Remove rows with missing values
full_data = full_data.dropna()

# Calculate percentage change and add it as a new column
full_data['↑   ↓   Change(%)'] = (
    (full_data['Predicted Price'] - full_data['Original Price']) / full_data['Original Price'] * 100
)

# Display the final table
st.subheader('Original Price VS Predicted Price')
st.dataframe(
    full_data.style.set_table_styles([
        {'selector': 'th:nth-child(1)', 'props': [('min-width', '150px')]},
        {'selector': 'th:nth-child(2)', 'props': [('min-width', '150px')]},
        {'selector': 'th:nth-child(3)', 'props': [('min-width', '150px')]},
        {'selector': 'th:nth-child(4)', 'props': [('min-width', '200px')]}
    ]),
    height=500,  # Adjust table height
    width=900   # Adjust table width
)

st.markdown('<hr>', unsafe_allow_html=True)