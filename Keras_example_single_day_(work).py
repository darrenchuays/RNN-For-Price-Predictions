# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:39:43 2019

@author: ali-ntu-016
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
asset=['DBS']
look_back = 30
# Importing the training set
dataset_train = pd.read_csv('train_set.csv').set_index('Date')[asset]
training_set = dataset_train.values #reshape(-1,1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(look_back, len(training_set)-1):
    X_train.append(training_set_scaled[i-look_back:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
net = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
net.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train.shape[1], 1)))
net.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
net.add(LSTM(units = 64, return_sequences = True))
net.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
net.add(LSTM(units = 64, return_sequences = True))
net.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
net.add(LSTM(units = 64))
net.add(Dropout(0.2))

# Adding the output layer
net.add(Dense(units = 1))

# Compiling the RNN
net.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
net.fit(X_train, y_train, epochs = 50, batch_size = 64)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('test_set.csv').set_index('Date')[asset]
mov_avg = dataset_test.rolling(window=15).mean()
mov_avg = mov_avg.fillna(0).values[look_back+1:].reshape(-1,1)
real_stock_price = dataset_test.values.reshape(-1,1)

# Getting the predicted stock price of 2017

real_stock_price = sc.transform(real_stock_price)
X_test = []
Y_test = []
for i in range(look_back, real_stock_price.shape[0]-1):
    X_test.append(real_stock_price[i-look_back:i, 0])
    Y_test.append(real_stock_price[i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
Y_test = np.array(Y_test).reshape(-1,1)
predicted_stock_price = net.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
true_stock_price = sc.inverse_transform(Y_test)

# Visualising the results
plt.plot(true_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.plot(mov_avg, color = 'green', label = 'SMA Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


