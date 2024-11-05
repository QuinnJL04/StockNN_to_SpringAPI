import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
import os

names = ['NVDA', 'AMZN', 'GOOGL', 'MSFT', 'AAPL']
types = ['']

def choose_stock(data, name:str, type:str):
    print(data.keys())
    print(data.head(5))
    return data[(type, name)]

def process_data():
    dirname = os.getcwd()
    csv_path = os.path.join(dirname, 'stock_data.csv')
    data = pd.read_csv(csv_path)
    return data
    #return data

def partition_data(data):
    train_size = int(len(data)*0.65)
    test_size = len(data) - train_size
    train_data,test_data = data[0:train_size,:],data[train_size:len(data),:1]
if __name__ == "__main__":
    data = process_data()
    nvda_close = choose_stock(data, 'NVDA', 'CLOSE')
    partition_data(nvda_close)