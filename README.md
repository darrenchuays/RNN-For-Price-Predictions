# RNN-For-Price-Predictions
I used Pytorch/Keras LSTM to predict future prices for future use in portfolio allocation.

Combined.csv contains the adjusted closing price of MCT, DBS, STEng and SIA
Deterministic:

-Single Day prediction looks back a period to predict the next day price (e.g. price from Day 1-30 to predict Day 31)

-Multi Day prediction looks back a series of periods to predict the next period price (e.g. price from Day 1-10, 11-20, 21-30 to predict 31-40)

Stochastic:

-Single Day MDN prediction looks back a period to predict the next day price by predicting parameters of its Gaussian MDN (e.g. price from Day 1-30 to predict Day 31)

-Multi Day MDN prediction looks back a period to predict the next day price by predicting parameters of its Gaussian MDN (e.g. price from Day 1-10, 11-20, 21-30 to predict 31-40)
