import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import yfinance as yf
import math
import matplotlib.pyplot as plt
import joblib

# Data Preprocessing
start_date = '2020-01-01'
end_date = '2024-01-01'

new_start_date = datetime.strptime(start_date, '%Y-%m-%d')
new_start_date = new_start_date - timedelta(days=80)
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

X = data[['Close', '10_MA', '20_MA', 'Volume', 'MACD', 'Signal Line', 'MACD Histogram']]
y = data['Change_3day']

# Split the dataset into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize a random forest regressor and train it with the training data
rf_model = RandomForestRegressor(random_state=1234)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, 'Trained_Models/RandomForest_model.h5')

# Make prediction of training data 
y_train_rf_pred = rf_model.predict(X_train)

# Evaluate performance
print("Training Data:")
mse_train = mean_squared_error(y_train, y_train_rf_pred)
mae_train = mean_absolute_error(y_train, y_train_rf_pred)
print(f'Mean Squared Error: {mse_train:.5f}')
print(f'Mean Absolute Error: {mae_train:.5f}')

# Make prediction of testing data 
y_test_rf_pred = rf_model.predict(X_test)

# Evaluate performance
print("\nTesting Data: ")
mse_test = mean_squared_error(y_test, y_test_rf_pred)
mae_test = mean_absolute_error(y_test, y_test_rf_pred)
print(f'Mean Squared Error: {mse_test:.5f}')
print(f'Mean Absolute Error: {mae_test:.5f}')



# Data Visualization
plt.figure(figsize=(12, 6))

# Price Prediction Calculation (Training Data)
train_close_price = data['Close']
train_close_price = train_close_price.to_numpy().flatten()
train_close_price = train_close_price[:len(y_train_rf_pred)]
train_pred_price = train_close_price + y_train_rf_pred

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
test_close_price = test_close_price[len(y_train_rf_pred):]
test_pred_price = test_close_price + y_test_rf_pred

# Scatter Plot (Testing Data)
pred_price = test_pred_price[:-3]
real_price = test_close_price[3:]
plt.subplot(1, 2, 2)
plt.scatter(real_price, pred_price, alpha=0.5, color='green', label='Predictions')
plt.plot([real_price.min(), real_price.max()], [real_price.min(), real_price.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Price Prediction (Testing Data)')
plt.legend()


# Plot for Random Forest predictions
plt.figure(figsize=(18, 6))

# Original Price
plt.plot(data.index, data.Close, label="Original Price", color="b")

# Training Data Prediction
plt.plot(data.index[:len(X_train)]+ pd.Timedelta(days=3), train_pred_price, label="Training data Prediction", color="g")

# Testing Data Prediction
plt.plot(data.index[len(X_train):]+ pd.Timedelta(days=3), test_pred_price, label="Testing data Prediction", color="brown")

plt.legend()
plt.title("Random Forest Prediction - Apple")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.tight_layout()
plt.show()