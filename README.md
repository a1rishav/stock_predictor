
## Stock predicitons using Regression

Yeah I know, stock prediction using **regression** can shrink eyebrows, but hold on and take a look at the prediction plots and accuracy score. It's going to blow your minds.

## Why Regression?

It's accuracy is appreciable and it's easy to implement.

## Can I use to trade stocks in real time ?

To trade stocks you need to perform actions like buy, hold and sell. You can dry Reinforcement learning algorithms for that like look at this super cool blog https://towardsdatascience.com/aifortrading-2edd6fac689d

## How to use this code?

- Download historical stock data from https://in.finance.yahoo.com/quote/TATASTEEL.NS/history?p=TATASTEEL.NS
- Save it to data directory
- Open stock_predictor.py and change the path to downloaded stock data
- Run stock_predictor.py
- Take a look at the scores of algos used
- Prediction plot is saved in the predictions directory

## Ready for goosebumbs?

Ridge Score : 0.9644882450009289
Lasso Score : 0.9632919774209426
Gradient Boost Score : 0.9293716359486464
ARD Score : 0.9655156746242711

![alt text](https://github.com/a1rishav/stock_predictor/blob/master/predictions/TATA_STEEL-predictions.png)

