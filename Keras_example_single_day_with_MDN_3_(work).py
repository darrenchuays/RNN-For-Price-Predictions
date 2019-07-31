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
training_set = dataset_train.values.reshape(-1,1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(look_back, len(training_set)-1):
    X_train.append(training_set_scaled[i-look_back:i, 0])
    y_train.append(training_set_scaled[i-look_back+1:i+1, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.layers import Dropout
from keras import backend as K
import tensorflow as tf

gausDim = 3
hDim = 64
# Initialising the RNN
rnn_x = Input(shape=(None, 1))
lstm = LSTM(hDim, return_sequences=True, recurrent_dropout=0.3)      
lstm1 = lstm(rnn_x)
lstm_output_model = LSTM(hDim, return_sequences=True, recurrent_dropout=0.3)(lstm1)
mdn = Dense(units = gausDim*3) 
mdn2 = mdn(lstm_output_model)
mdn_model = Dense(units = gausDim*3)(mdn2)  
model = Model(rnn_x, mdn_model)

print(model.summary())
# Customize loss function

def loss_func(y_true, y_pred):
    
    Pis = K.softmax(y_pred[:,:,:gausDim], axis=2)
    sigmas2 = K.exp(y_pred[:,:,2*gausDim:])
    log_norms = -0.5*K.log(2*np.pi*sigmas2) - (y_true - y_pred[:,:,gausDim:2*gausDim])**2/(2*sigmas2)
    norms = K.exp(log_norms)
    loss_final = K.sum(Pis*norms, axis=2)
    loss_final = K.mean(-K.log(loss_final))
    return loss_final

# Compiling the RNN
model.compile(optimizer = 'adam', loss = loss_func)

# Fitting the RNN to the Training set
loss = model.fit(X_train, y_train, epochs = 100, batch_size = 64)
loss_history = loss.history["loss"]


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
gausParam = model.predict(X_test)[:,-1,:]


def simple_sample(x):
    ks = np.zeros(x.shape[0])
    for i in range(int(x.shape[0])):
        ks[i]=np.random.choice(x.shape[1],1,p=x[i])
    return ks.astype(int)

def predict_price(params, dim):
    sampled = np.zeros(params.shape[0])
    Pis = K.get_value(K.softmax(params[:,:dim], axis=1))
    mus = params[:,dim:dim*2]
    sigmas = K.get_value(K.sqrt(K.exp(params[:,dim*2:])))
    ks = simple_sample(Pis)
    for i in range(params.shape[0]):
        stdnorm = np.random.randn()
        sampled[i]=(stdnorm * sigmas[i,ks[i]] + mus[i,ks[i]])
    return sampled
predicted_stock_price =[]
tries = 3
for j in range(3):
    temp = predict_price(gausParam, gausDim).reshape(-1,1)
    temp = sc.inverse_transform(temp)
    predicted_stock_price.append(temp)

predicted_stock_price = np.array(predicted_stock_price)
mean_predicted_stock_price = np.mean(predicted_stock_price, axis = 0)
true_stock_price = sc.inverse_transform(Y_test)

# Visualising the results
plt.plot(true_stock_price, color = 'red', label = 'Real', linewidth = 4)
for i in range(3):
    plt.plot(predicted_stock_price[i], label = 'Predicted' + str(i))
plt.plot(mov_avg, color = 'green', label = 'SMA')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
plt.plot(loss_history)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.show()
plt.plot(true_stock_price, color = 'red', label = 'Real')
plt.plot(mean_predicted_stock_price, color = 'black', label = 'Mean Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(loc = 1)
plt.show()