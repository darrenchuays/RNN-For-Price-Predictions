# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:44:29 2019

@author: Darren
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

asset=['DBS']
bsz = 128
epochs = 1500
seqlen = 3
length = 10
z_size = 10
look_back = seqlen*length
# Importing data
dataset_train = pd.read_csv('train_set.csv').set_index('Date')[asset]
training_set = dataset_train.values.reshape(-1,1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Setting up input and target
X_train = []
Y_train = []
for i in range(look_back, len(training_set)-length):
    temp = training_set_scaled[i-look_back:i, 0].reshape(seqlen, length)
    temp2 = training_set_scaled[i-look_back+length:i+length, 0].reshape(seqlen, length)
    X_train.append(temp)
    Y_train.append(temp2)
X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], length))
Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1], length))

n_hidden = 256
n_gaussians = 7
z = torch.Tensor(X_train)
targets = torch.Tensor(Y_train)

def detach(states):
    return [state.detach() for state in states]

class MDNRNN(nn.Module):
    def __init__(self, z_size, n_hidden=n_hidden, n_gaussians=n_gaussians, n_layers=2):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(z_size, n_hidden, n_layers, batch_first=True, dropout=0.4)
        self.fc1 = nn.Linear(n_hidden, n_gaussians)
        self.fc2 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc3 = nn.Linear(n_hidden, n_gaussians)
        
    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.n_gaussians)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians)
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
        
        
    def forward(self, x, h):
        # Forward propagate LSTM
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
    def init_hidden(self, bszh):
        return (torch.zeros(self.n_layers, bszh, self.n_hidden), torch.zeros(self.n_layers, bszh, self.n_hidden))
           
model = MDNRNN(z_size, n_hidden)

def mdn_loss_fn(y, pi, mu, sigma):
    m = torch.distributions.Normal(loc=mu, scale= sigma.unsqueeze(3).expand_as(mu) )
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * (pi.unsqueeze(3).expand_as(mu)), dim=2)
    loss = -torch.log(loss)
    return loss.mean()

def criterion(y, pi, mu, sigma):
    y = y.unsqueeze(2).expand_as(mu)
    return mdn_loss_fn(y, pi, mu, sigma)

optimizer = torch.optim.Adam(model.parameters())


loss_history = []

for epoch in range(epochs):
    # Set initial hidden and cell states
    hidden = model.init_hidden(bsz)

    # Get mini-batch inputs and targets
    indx = np.random.randint(X_train.shape[0], size=bsz)
    inputs = z[indx,:,:]
    target = targets[indx,:]
    
    # Forward pass
    hidden = detach(hidden)
    (pi, mu, sigma), hidden = model.forward(inputs, hidden)
    loss = criterion(target, pi, mu, sigma)
    
    # Backward and optimize
    model.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss)

    print (epoch, loss)


# Getting the real stock price
dataset_test = pd.read_csv('test_set.csv').set_index('Date')[asset]
mov_avg = dataset_test.rolling(window=15).mean()
mov_avg = mov_avg.fillna(0).values[look_back+1:].reshape(-1,1)  #Simple Moving Average
real_stock_price = dataset_test.values.reshape(-1,1)

# Getting the predicted stock price of 2017

real_stock_price = sc.transform(real_stock_price)
X_test = []
Y_test = []

for i in range(look_back, real_stock_price.shape[0]-length, length):
    temp = real_stock_price[i-look_back:i, 0].reshape(seqlen, length)
    temp2 = real_stock_price[i:i+length, 0].reshape(-1,1)
    X_test.append(temp)
    Y_test.append(temp2)

   
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], length))
Y_test = np.array(Y_test).reshape(-1,1)
X_test = torch.Tensor(X_test)
hidden = model.init_hidden(int(X_test.shape[0]))
hidden = detach(hidden)
gausParam, _ = model.forward(X_test, hidden)

def simple_sample(x):
    ks = np.zeros(x.shape[0])
    for i in range(int(x.shape[0])):
        ks[i]=np.random.choice(x.shape[1],1,p=x[i])
    return ks.astype(int)

def predict_price(params, dim):
    sampled = np.zeros((params[0].shape[0],params[1].shape[-1]))
    Pis = (params[0][:,-1,:]).detach().numpy()
    mus = (params[1][:,-1,:,:]).detach().numpy()
    sigmas = (params[2][:,-1,:]).detach().numpy()
    ks = simple_sample(Pis)
    for i in range(params[0].shape[0]):
        stdnorm = np.random.randn()
        sampled[i]=(stdnorm * sigmas[i,ks[i]] + mus[i,ks[i],:])
    return sampled

predicted_stock_price =[]
tries = 3
for j in range(tries):
    temp = predict_price(gausParam, n_gaussians).reshape(-1,1)
    temp = sc.inverse_transform(temp)
    predicted_stock_price.append(temp)

predicted_stock_price = np.array(predicted_stock_price)
mean_predicted_stock_price = np.mean(predicted_stock_price, axis = 0)
true_stock_price = sc.inverse_transform(Y_test)

# Visualising the results
plt.plot(true_stock_price, color = 'red', label = 'Real')
for i in range(tries):
    plt.plot(predicted_stock_price[i], label = 'Predicted' + str(i))
    
plt.plot(mov_avg, color = 'green', label = 'SMA')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(loc = 1)
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
   

