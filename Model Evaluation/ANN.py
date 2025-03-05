import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

import keras
from keras import layers
from keras import models

from sklearn.metrics import mean_squared_error, r2_score

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

# Initializing the model
regressorANN = keras.Sequential()
# Adding input layer and first hidden layer
regressorANN.add(layers.Dense(units=50, activation='relu', input_shape=(50, 7)))
# Adding a second hidden layer
regressorANN.add(layers.Dense(units=25, activation='relu'))
# Flatten the output to ensure a single output per sample
regressorANN.add(layers.Flatten())
# Adding the output layer
regressorANN.add(layers.Dense(units=1))
# Compiling the model
regressorANN.compile(optimizer='adam', loss='mean_squared_error', metrics=["mae"])
# Storing mse and mae values
ANNmse = regressorANN.evaluate(X_test, y_test, verbose=0)
# Fitting the model
regressorANN.fit(X_train, y_train, batch_size=1, epochs=12)
regressorANN.summary()


print(ANNmse)

# Make prediction of training data 
y_train_ANN_pred = regressorANN.predict(X_train)
y_train_ANN_pred = scaler.inverse_transform(y_train_ANN_pred)
y_train = scaler.inverse_transform(y_train)

# Evaluate performance
print("Training Data:")
mse = mean_squared_error(y_train, y_train_ANN_pred)
r2 = r2_score(y_train, y_train_ANN_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Make prediction of testing data 
y_test_ANN_pred = regressorANN.predict(X_test)
y_test_ANN_pred = scaler.inverse_transform(y_test_ANN_pred)
y_test = scaler.inverse_transform(y_test)

# Evaluate performance
print("\nTesting Data: ")
mse_test = mean_squared_error(y_test, y_test_ANN_pred)
r2_test = r2_score(y_test, y_test_ANN_pred)
print(f'Mean Squared Error: {mse_test:.2f}')
print(f'R-squared: {r2_test:.2f}')



# Data Visualization
plt.figure(figsize=(12, 6))

# Scatter plot (Training Data)
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_ANN_pred, alpha=0.5, color='blue', label='Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual Changes')
plt.ylabel('Predicted Changes')
plt.title('Price Change Prediction (Training Data)')
plt.legend()

# Scatter Plot (Testing Data)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_ANN_pred, alpha=0.5, color='green', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual Changes')
plt.ylabel('Predicted Changes')
plt.title('Price Change Prediction (Testing Data)')
plt.legend()


# Price Prediction Calculation (Training Data)
train_close_price = data['Close']
train_close_price = train_close_price.to_numpy().flatten()
train_close_price = train_close_price[49:len(y_train_ANN_pred)+49]
y_train_ANN_pred = y_train_ANN_pred.reshape(y_train_ANN_pred.shape[0])
train_pred_price = train_close_price + y_train_ANN_pred


# Price Prediction Calculation (Testing Data)
test_close_price = data['Close']
test_close_price = test_close_price.to_numpy().flatten()
test_close_price = test_close_price[len(y_train_ANN_pred)+49+49:]
y_test_ANN_pred = y_test_ANN_pred.reshape(y_test_ANN_pred.shape[0])
test_pred_price = test_close_price + y_test_ANN_pred


# Plot for Random Forest predictions
plt.figure(figsize=(18, 6))

# Original Price
plt.plot(data.index, data.Close, label="Original Price", color="b")

# Training Data Prediction
plt.plot(data.index[49:len(X_train)+49]+ pd.Timedelta(days=3), train_pred_price, label="Training data Prediction", color="g")

# Testing Data Prediction
plt.plot(data.index[len(X_train)+49+49:]+ pd.Timedelta(days=3), test_pred_price, label="Testing data Prediction", color="brown")

plt.legend()
plt.title("ANN Prediction - Apple")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.tight_layout()
plt.show()