import yfinance as yf
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
import math
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


dirname = os.getcwd()
csv_path = os.path.join(dirname, 'stock_data.csv')
scaler = MinMaxScaler()


def get_features():
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

names = ['NVDA', 'AMZN', 'GOOGL', 'MSFT', 'AAPL']
types = ['']
def choose_stock(data, name:str, type:str):
    new_data = data.reset_index()[(type, name)]
    return new_data


def partition_data(data):
    train_size = int(len(data)*0.65)
    test_size = len(data) - train_size
    train_data,test_data = data[0:train_size,:],data[train_size:len(data),:1]
    return train_data,test_data


def create_dataset(dataset, time_step = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
                   a = dataset[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append(dataset[i + time_step,0])
    return np.array(dataX),np.array(dataY)

def clean_dataset(X_train, X_test, Y_train, Y_test):
     # Ensure the data size is divisible by 100 for reshaping
    train_samples = len(X_train) - (len(X_train) % 100)  # Truncate to the closest multiple of 100 cause input size was too large before
    test_samples = len(X_test) - (len(X_test) % 100) 

    X_train_truncated = X_train[:train_samples]
    X_test_truncated = X_test[:test_samples]

    X_train_tensor = torch.tensor(X_train_truncated, dtype=torch.float32).reshape(-1, 100, 1)
    X_test_tensor = torch.tensor(X_test_truncated, dtype=torch.float32).reshape(-1, 100, 1)

    Y_train_tensor = torch.tensor(Y_train[:train_samples], dtype=torch.float32).reshape(-1, 1)
    Y_test_tensor = torch.tensor(Y_test[:test_samples], dtype=torch.float32).reshape(-1, 1)
    return X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, time_step):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.fc(x[:, -1, :])  
        return x

def train_model(X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor):
     # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_size = 1  
    hidden_layer_size = 50
    model = LSTMModel(input_size, hidden_layer_size, time_step)

    print(f"Using device: {device}")
    model = model.to(device)

    # Move data to GPU
    X_train_tensor = X_train_tensor.to(device)
    Y_train_tensor = Y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    Y_test_tensor = Y_test_tensor.to(device)
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)  # Move batch to GPU
            Y_batch = Y_batch.to(device)  # Move target to GPU

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Validation loop
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)  # Move batch to GPU
            Y_batch = Y_batch.to(device)  # Move target to GPU

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            print(f"Test Loss: {loss.item()}")

    model.save('my_lstm_model.h5')


if __name__ == '__main__':
    data = get_features()

    this_data = choose_stock(data, 'NVDA', 'Close')
    this_data = scaler.fit_transform(np.array(data).reshape(-1,1)) 
    train_data, test_data = partition_data(this_data)
    print(train_data[0:5])

    time_step = 100
    X_train,Y_train =  create_dataset(train_data,time_step)
    X_test,Y_test =  create_dataset(test_data,time_step)      
    X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor = clean_dataset(X_train, X_test, Y_train, Y_test)  
    train_model(X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor)
