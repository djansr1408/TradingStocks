# Trading Stocks using LSTM
This project is about predicting stock prices using LSTM neural network and implementing simple trading strategy.

## Introduction

Developing trading strategy is quite challenging task because of variety of approaches that could be applied. Some of the mostly used are: momentum strategy, reversion strategy and forecasting strategy.
For this assignment, I will use forecasting strategy combined with deep learning. The main idea is to predict the next value of a stock based on some historical factors (data). 


## Data

In this case, our historical data are prices of a SPY stock from 1st Jan 2010 to 31st Dec 2015. This data contains information for opening, closing, highest and lowest prices for each day, as presented on the Figure 1. In this project, only CLOSE prices were used for prediction and the graphic is showed on Figure 2. This data contains 4025 days for the given period.

