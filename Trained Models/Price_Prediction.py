import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import keras

# Data Preprocessing
end_date = datetime.now()
start_date = end_date - timedelta(days=730) # 2 years
new_start_date = start_date - timedelta(days=60)

stock = "AMZN"
data = yf.download(stock, new_start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

# Feature Calculation
data['10_MA'] = data['Close'].rolling(window=10).mean()
data['20_MA'] = data['Close'].rolling(window=20).mean()
data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD Histogram'] = data['MACD'] - data['Signal Line']

data['Change_3day'] = data['Close'].shift(-3) - data['Close']

# Drop additional rows 
data = data[data.index >= start_date]

# Linear Normalization 
dataset_Close = data['Close'].to_numpy()
dataset_10MA = data['10_MA'].to_numpy()
dataset_20MA = data['20_MA'].to_numpy()
dataset_volume = data['Volume'].to_numpy()
dataset_MACD = data['MACD'].to_numpy()
dataset_SignalLine = data['Signal Line'].to_numpy()
dataset_MACDHistogram = data['MACD Histogram'].to_numpy()
dataset_Change3day = data['Change_3day'].to_numpy()

# Reshape
dataset_Close = np.reshape(dataset_Close, (-1, 1))
dataset_10MA = np.reshape(dataset_10MA, (-1, 1))
dataset_20MA = np.reshape(dataset_20MA, (-1, 1))
dataset_volume = np.reshape(dataset_volume, (-1, 1))
dataset_MACD = np.reshape(dataset_MACD, (-1, 1))
dataset_SignalLine = np.reshape(dataset_SignalLine, (-1, 1))
dataset_MACDHistogram = np.reshape(dataset_MACDHistogram, (-1, 1))
dataset_Change3day = np.reshape(dataset_Change3day, (-1, 1))


# Scaling
training_data_len = math.ceil(len(data) * .8)
scaler = MinMaxScaler(feature_range=(0,1))

scaler.fit(dataset_Close[:training_data_len])
dataset_Close = scaler.transform(dataset_Close)

scaler.fit(dataset_10MA[:training_data_len])
dataset_10MA = scaler.transform(dataset_10MA)

scaler.fit(dataset_20MA[:training_data_len])
dataset_20MA = scaler.transform(dataset_20MA)

scaler.fit(dataset_volume[:training_data_len])
dataset_volume = scaler.transform(dataset_volume)

scaler.fit(dataset_MACD[:training_data_len])
dataset_MACD = scaler.transform(dataset_MACD)

scaler.fit(dataset_SignalLine[:training_data_len])
dataset_SignalLine = scaler.transform(dataset_SignalLine)

scaler.fit(dataset_MACDHistogram[:training_data_len])
dataset_MACDHistogram = scaler.transform(dataset_MACDHistogram)

scaler.fit(dataset_Change3day[:training_data_len])
dataset_Change3day = scaler.transform(dataset_Change3day)


X_past = []
y_past = []

X_future = []


for i in range(49, dataset_Close.shape[0]-3):
    # X 
    timesteps = []
    for j in range(i-49, i+1):
        features = []
        features.append([
            dataset_Close[j][0],    
            dataset_10MA[j][0],       
            dataset_20MA[j][0],        
            dataset_volume[j][0],   
            dataset_MACD[j][0],    
            dataset_SignalLine[j][0], 
            dataset_MACDHistogram[j][0]
        ])
        timesteps.append(features)
    X_past.append(timesteps)
    # y
    y_past.append(dataset_Change3day[i][0])


for i in range(dataset_Close.shape[0]-3, dataset_Close.shape[0]):
    # X 
    timesteps = []
    for j in range(i-49, i+1):
        features = []
        features.append([
            dataset_Close[j][0],    
            dataset_10MA[j][0],       
            dataset_20MA[j][0],        
            dataset_volume[j][0],   
            dataset_MACD[j][0],    
            dataset_SignalLine[j][0], 
            dataset_MACDHistogram[j][0]
        ])
        timesteps.append(features)
    X_future.append(timesteps)


X_past = np.array(X_past)
X_past = np.reshape(X_past, (X_past.shape[0], X_past.shape[1],7))

y_past = np.array(y_past)
y_past = np.reshape(y_past, (y_past.shape[0], 1))

X_future = np.array(X_future)
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1],7))


# ANN Prediction
ANN_model = keras.models.load_model('Trained Models/ANN_model.h5')

# Past prediction
y_past_predict = ANN_model.predict(X_past)
y_past_predict = scaler.inverse_transform(y_past_predict)
y_past_predict = y_past_predict.reshape(y_past_predict.shape[0])

past_predict_price = data['Close'].to_numpy().flatten()
past_predict_price = past_predict_price[49:-3]
past_predict_price = past_predict_price + y_past_predict

# Future prediction
y_future_predict = ANN_model.predict(X_future)
y_future_predict = scaler.inverse_transform(y_future_predict)
y_future_predict = y_future_predict.reshape(y_future_predict.shape[0])

future_predict_price = data['Close'].to_numpy().flatten()
future_predict_price = future_predict_price[-3:]
future_predict_price = future_predict_price + y_future_predict

future_predict_price = np.insert(future_predict_price, 0, data['Close'].tail(1).values[0])


print("\n\nANN: ")
print(f'Latest Closing Price: {future_predict_price[0]:.2f}')
print(f'Future Closing Price (Next Day): {future_predict_price[1]:.2f}')
print(f'Future Closing Price (Two Days Ahead): {future_predict_price[2]:.2f}')
print(f'Future Closing Price (Three Days Ahead): {future_predict_price[3]:.2f}\n')

# Plot the Graph
plt.figure(figsize=(18, 6))
plt.plot(data.index, data.Close, label="Original Price", color="b") # Original Price

plt.plot(data.index[52:-30], past_predict_price[:-30], label="Past data Prediction", color="brown") # Past Prediction Price

last_index = data.index[-1]
future_index = pd.date_range(start=last_index, periods=len(future_predict_price), freq='D')
plt.plot(future_index, future_predict_price, label="Future data Prediction", color="r") # Future Prediction Price

plt.legend()
plt.title(f'ANN Prediction - {stock}')
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.tight_layout()



# LSTM Prediction
LSTM_model = keras.models.load_model('Trained Models/LSTM_model.h5')

# Past prediction
y_past_predict = LSTM_model.predict(X_past)
y_past_predict = scaler.inverse_transform(y_past_predict)
y_past_predict = y_past_predict.reshape(y_past_predict.shape[0])

past_predict_price = data['Close'].to_numpy().flatten()
past_predict_price = past_predict_price[49:-3]
past_predict_price = past_predict_price + y_past_predict

# Future prediction
y_future_predict = LSTM_model.predict(X_future)
y_future_predict = scaler.inverse_transform(y_future_predict)
y_future_predict = y_future_predict.reshape(y_future_predict.shape[0])

future_predict_price = data['Close'].to_numpy().flatten()
future_predict_price = future_predict_price[-3:]
future_predict_price = future_predict_price + y_future_predict
future_predict_price = np.insert(future_predict_price, 0, data['Close'].tail(1).values[0])

print("\n\nLSTM: ")
print(f'Latest Closing Price: {future_predict_price[0]:.2f}')
print(f'Future Closing Price (Next Day): {future_predict_price[1]:.2f}')
print(f'Future Closing Price (Two Days Ahead): {future_predict_price[2]:.2f}')
print(f'Future Closing Price (Three Days Ahead): {future_predict_price[3]:.2f}\n')

# Plot the Graph
plt.figure(figsize=(18, 6))

plt.plot(data.index, data.Close, label="Original Price", color="b") # Original Price

plt.plot(data.index[52:-30], past_predict_price[:-30], label="Past data Prediction", color="brown") # Past Prediction Price

last_index = data.index[-1]
future_index = pd.date_range(start=last_index, periods=len(future_predict_price), freq='D')
plt.plot(future_index, future_predict_price, label="Future data Prediction", color="r") # Future Prediction Price

plt.legend()
plt.title(f'LSTM Prediction - {stock}')
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.tight_layout()


plt.show()
