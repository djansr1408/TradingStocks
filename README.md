# Trading Stocks using LSTM
This project is about predicting stock prices using LSTM neural network and implementing simple trading strategy.

## Introduction

Developing trading strategy is quite challenging task because of variety of approaches that could be applied. Some of the mostly used are: momentum strategy, reversion strategy and forecasting strategy.
For this assignment, I will use forecasting strategy combined with deep learning. The main idea is to predict the next value of a stock based on some historical factors (data). 


## Data

In this case, our historical data are prices of a SPY stock from 1st Jan 2010 to 31st Dec 2015. This data contains information for opening, closing, highest and lowest prices for each day, as presented on the Figure 1. In this project, only CLOSE prices were used for prediction and the graphic is showed on Figure 2. This data contains 4025 days for the given period.

<p align="center">
<img style="float: center;" align="center" src="./images/data.PNG" >
Figure 1: Data for period from 1st Jan 2010 to 31st Dec 2015
</p>


<p align="center">
<img style="float: center;margin:0 auto; " align="center" src="./images/datagraphic.png">   
</p>

<p align="center">
Figure 2: CLOSE prices data
</p>


This data should be preprocessed so to put it into the right range. Normalization is problem here because trading data tends to change its statistics (mean, variance) over time, so we can’t simply subtract the mean from this whole dataset and divide it by its variance. Instead, we split the data into adjacent windows of size input_length, and values in each window divide with the value of last sample in previous window. After this, we get normalized data as shown on the Figure 3. On this figure, normalized data is averaged across windows where input_length=3. From this original data 90% is train data and the rest 10% is validation data.
