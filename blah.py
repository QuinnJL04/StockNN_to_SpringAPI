import yfinance as yf
import pandas as pd
import os
def get_features():

    """
    Gathers features for the given 5 stocks and drops any unnecessary columns
    """
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    data = yf.download(stocks, start="2018-01-01", end="2024-01-01", interval="1d")

    missing_data = data.isnull().sum()

    # Calculating the percentage change to see how a stock's price has increased or decreased in value
    for stock in stocks:
        # Use correct formatting
        data[('Pct Change', stock)] = data['Adj Close'][stock].pct_change()

    # Calculating moving averages to see trends without short-term fluctuations 
    # Calculate moving averages
    for stock in stocks:
        data[('MA_5', stock)] = data['Adj Close'][stock].rolling(window=5).mean()
        data[('MA_20', stock)] = data['Adj Close'][stock].rolling(window=20).mean()
        data[('MA_50', stock)] = data['Adj Close'][stock].rolling(window=50).mean()

    # Calculate volatility using the percentage change
    for stock in stocks:
        # Ensure the percentage change column exists before calculating volatility
        if ('Pct Change', stock) in data.columns:
            data[('Volatility', stock)] = data[('Pct Change', stock)].rolling(window=30).std()

    # Calculating the volume moving average and spikes
    for stock in stocks:
        data[('Volume MA', stock)] = data['Volume'][stock].rolling(window=20).mean()
        data[('Volume Spike', stock)] = data['Volume'][stock] > (data[('Volume MA', stock)] * 1.5)  # Days with volume significantly higher than mean

    # Drop any rows with NaN values resulting from the rolling calculations
    data.dropna(inplace=True)
    print(data.head(5))
    return data

data = get_features()
names = ['NVDA', 'AMZN', 'GOOGL', 'MSFT', 'AAPL']
types = ['Close', 'Open']
def choose_stock(data, type:str):
    """
    Allows user to chose which stock and what time of day to train the model on
    """
    nvda = data.reset_index()[(type, 'NVDA')]
    amzn = data.reset_index()[(type), 'AMZN']
    googl = data.reset_index()[(type), 'GOOGL']
    msft = data.reset_index()[(type), 'MSFT']
    appl = data.reset_index()[(type, 'AAPL')]
    combined_data = pd.concat([nvda, amzn, googl, msft,appl], axis=1)
    print(combined_data.head(5))
    return combined_data
choose_stock(data, 'Open')