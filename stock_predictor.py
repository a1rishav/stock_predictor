# Simple Linear Regression
# Importing the libraries
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import isna, isnull
from sklearn import datasets, linear_model
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ARDRegression


class StockPredictor:

    def load_data(self, stock_name, file_path):
        raw_df = pd.read_csv(file_path, usecols=["Date", "Adj Close"])
        raw_df.rename(columns={'Adj Close': 'adj_close', "Date": "date"}, inplace=True)
        raw_df['date'] = pd.to_datetime(raw_df['date'], format='%Y-%m-%d')
        raw_df.set_index('date', inplace=True)
        return raw_df

    def preprocess(self, raw_df):
        # remove null rows
        raw_df = raw_df[raw_df['adj_close'].notnull()]
        return raw_df

    def split(self, df, test_data_percentage=20):
        test_df_size = int(test_data_percentage * 0.01 * len(df))
        train_df = df[:-test_df_size]
        test_df = df[-test_df_size:]
        return train_df, test_df

    def plot_and_save(self, df, label, title, figsize=(16, 8)):
        figure = df.plot(label=label, figsize=figsize, title=title).get_figure()
        figure.savefig(os.path.join("predictions", label + "-" + title + ".png"))
        print()

    def prepare_training_data(self, df, days_to_look_back=32):
        num_samples = len(df) - days_to_look_back
        indices = np.reshape(np.arange(num_samples).astype(np.int)[:, None], (num_samples, 1)) + np.arange(
            days_to_look_back + 1).astype(np.int)
        data = df['adj_close'].values[indices]

        X = data[:, :-1]
        y = data[:, -1]

        split_fraction = 0.8
        ind_split = int(split_fraction * num_samples)
        X_train = X[:ind_split]
        y_train = y[:ind_split]
        X_test = X[ind_split:]
        y_test = y[ind_split:]

        return X_train, y_train, X_test, y_test

    def train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def predict(self, ridge_regressor, X_test):
        return ridge_regressor.predict(X_test)

    def print_scores(self, regressor_name, regressor, X_test, y_test):
        print(regressor_name + " Score : " + str(regressor.score(X_test, y_test)))


if __name__ == '__main__':
    stock_predictor = StockPredictor()
    raw_df = stock_predictor.load_data("TATA STEEL", "data/TATASTEEL.NS.csv")
    # raw_df = stock_predictor.load_data("TATA STEEL", "data/AAPL.csv")
    # raw_df = stock_predictor.load_data("TATA STEEL", "data/TCS.NS.csv")

    filtered_df = stock_predictor.preprocess(raw_df)

    days_to_look_back = 32
    X_train, y_train, X_test, y_test = stock_predictor.prepare_training_data(filtered_df, days_to_look_back)

    ridge_regressor = stock_predictor.train(Ridge(), X_train, y_train)
    lasso_regressor = stock_predictor.train(Lasso(), X_train, y_train)
    gradient_boost_regressor = stock_predictor.train(GradientBoostingRegressor(), X_train, y_train)
    ard_regressor = stock_predictor.train(ARDRegression(), X_train, y_train)

    ridge_predictions = stock_predictor.predict(ridge_regressor, X_test)
    lasso_predictions = stock_predictor.predict(lasso_regressor, X_test)
    gradient_boost_predictions = stock_predictor.predict(gradient_boost_regressor, X_test)
    ard_predictions = stock_predictor.predict(ard_regressor, X_test)

    stock_predictor.print_scores("Ridge", ridge_regressor, X_test, y_test)
    stock_predictor.print_scores("Lasso", lasso_regressor, X_test, y_test)
    stock_predictor.print_scores("Gradient Boost", gradient_boost_regressor, X_test, y_test)
    stock_predictor.print_scores("ARD", ard_regressor, X_test, y_test)

    filtered_df = filtered_df[len(X_train) + days_to_look_back:]
    filtered_df['date'] = filtered_df.index
    filtered_df.reset_index(drop=True, inplace=True)

    filtered_df["ridge_predictions"] = pd.Series(ridge_predictions)
    filtered_df["lasso_predictions"] = pd.Series(lasso_predictions)
    filtered_df["gradient_boost_predictions"] = pd.Series(gradient_boost_predictions)
    filtered_df["ard_predictions"] = pd.Series(ard_predictions)

    filtered_df.set_index(filtered_df['date'], inplace=True)
    filtered_df.drop(columns=["date"], inplace=True)
    stock_predictor.plot_and_save(filtered_df, "TATA_STEEL", "predictions")
