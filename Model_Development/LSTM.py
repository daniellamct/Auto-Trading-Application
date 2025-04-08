import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras
from keras import layers
from keras import models


# Data Preprocessing
start_date = '2022-01-01'
end_date = '2024-01-01'

new_start_date = datetime.strptime(start_date, '%Y-%m-%d')
new_start_date = new_start_date - timedelta(days=60)
new_start_date = new_start_date.strftime('%Y-%m-%d')

new_end_date = datetime.strptime(end_date, '%Y-%m-%d')
new_end_date = new_end_date + timedelta(days=3)
new_end_date = new_end_date.strftime('%Y-%m-%d')

data = yf.download("AAPL", new_start_date, new_end_date)

# Moving Average Calculation
data['10_MA'] = data['Close'].rolling(window=10).mean()
data['20_MA'] = data['Close'].rolling(window=20).mean()

# MACD Calculation
data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD Histogram'] = data['MACD'] - data['Signal Line']

# Change Calculation
data['Change_3day'] = data['Close'].shift(-3) - data['Close']

# Drop additional rows 
data = data[data.index >= start_date]
data = data[:-3]

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



X_train = []
X_test = []
y_train = []
y_test = []


for i in range(49, training_data_len):
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
    X_train.append(timesteps)
    # y
    y_train.append(dataset_Change3day[i][0])

for i in range(training_data_len+49, len(data)):
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
    X_test.append(timesteps)
    # y
    y_test.append(dataset_Change3day[i][0])

X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],7))

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],7))

y_train = np.array(y_train)
y_train = np.reshape(y_train, (y_train.shape[0], 1))

y_test = np.array(y_test)
y_test = np.reshape(y_test, (y_test.shape[0], 1))

# Initialising the LSTM model
regressorLSTM = models.Sequential()
#Adding LSTM layers
regressorLSTM.add(layers.LSTM(50, return_sequences = True, input_shape = (X_train.shape[1], 7)))
regressorLSTM.add(layers.LSTM(50, return_sequences = True))
regressorLSTM.add(layers.LSTM(50, return_sequences = False))
regressorLSTM.add(layers.Dense(25))
#Adding the output layer
regressorLSTM.add(layers.Dense(1))
#Compiling the model
regressorLSTM.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ["mae"])
#Fitting the model
regressorLSTM.fit(X_train, y_train, batch_size = 5, epochs = 200)
LSTMmse = regressorLSTM.evaluate(X_test, y_test)
regressorLSTM.summary()
print(LSTMmse)

# Save the trained model
regressorLSTM.save('Trained_Models/LSTM_model.h5')

# Make prediction of training data 
y_train_LSTM_pred = regressorLSTM.predict(X_train)

# Evaluate performance
print("Training Data:")
mse_train = mean_squared_error(y_train, y_train_LSTM_pred)
mae_train = mean_absolute_error(y_train, y_train_LSTM_pred)
print(f'Mean Squared Error: {mse_train:.5f}')
print(f'Mean Absolute Error: {mae_train:.5f}')


# Make prediction of testing data 
y_test_LSTM_pred = regressorLSTM.predict(X_test)

# Evaluate performance
print("\nTesting Data: ")
mse_test = mean_squared_error(y_test, y_test_LSTM_pred)
mae_test = mean_absolute_error(y_test, y_test_LSTM_pred)
print(f'Mean Squared Error: {mse_test:.5f}')
print(f'Mean Absolute Error: {mae_test:.5f}')
print("Testing inversed: ")
mse_test = mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(y_test_LSTM_pred))
mae_test = mean_absolute_error(scaler.inverse_transform(y_test), scaler.inverse_transform(y_test_LSTM_pred))
print(f'Mean Squared Error: {mse_test:.5f}')
print(f'Mean Absolute Error: {mae_test:.5f}')

# Convert back to original scale for data visualization
y_train_LSTM_pred = scaler.inverse_transform(y_train_LSTM_pred)
y_test_LSTM_pred = scaler.inverse_transform(y_test_LSTM_pred)

# Data Visualization
plt.figure(figsize=(12, 6))

# Price Prediction Calculation (Training Data)
train_close_price = data['Close']
train_close_price = train_close_price.to_numpy().flatten()
train_close_price = train_close_price[49:len(y_train_LSTM_pred)+49]
y_train_LSTM_pred = y_train_LSTM_pred.reshape(y_train_LSTM_pred.shape[0])
train_pred_price = train_close_price + y_train_LSTM_pred

# Scatter plot (Training Data)
pred_price = train_pred_price[:-3]
real_price = train_close_price[3:]
plt.subplot(1, 2, 1)
plt.scatter(real_price, pred_price, alpha=0.5, color='blue', label='Predictions')
plt.plot([real_price.min(), real_price.max()], [real_price.min(), real_price.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Price Prediction (Training Data)')
plt.legend()


# Price Prediction Calculation (Testing Data)
test_close_price = data['Close']
test_close_price = test_close_price.to_numpy().flatten()
test_close_price = test_close_price[len(y_train_LSTM_pred)+49+49:]
y_test_LSTM_pred = y_test_LSTM_pred.reshape(y_test_LSTM_pred.shape[0])
test_pred_price = test_close_price + y_test_LSTM_pred

# Scatter plot (Training Data)
pred_price = test_pred_price[:-3]
real_price = test_close_price[3:]
plt.subplot(1, 2, 2)
plt.scatter(real_price, pred_price, alpha=0.5, color='green', label='Predictions')
plt.plot([real_price.min(), real_price.max()], [real_price.min(), real_price.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Price Prediction (Testing Data)')
plt.legend()



# Plot for LSTM
plt.figure(figsize=(18, 6))

# Original Price
plt.plot(data.index, data.Close, label="Original Price", color="b")

# Training Data Prediction
plt.plot(data.index[49:len(X_train)+49]+ pd.Timedelta(days=3), train_pred_price, label="Training data Prediction", color="g")

# Testing Data Prediction
plt.plot(data.index[len(X_train)+49+49:]+ pd.Timedelta(days=3), test_pred_price, label="Testing data Prediction", color="brown")

plt.legend()
plt.title("LSTM Prediction - Apple")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.tight_layout()
plt.show()