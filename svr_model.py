import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


def build_svr_model(df):
    """
    df must contain columns: Date, Close
    Uses last 60 rows for training.
    """

    df = df.copy()
    df = df.dropna()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df = df.tail(60)

    if len(df) < 20:
        raise ValueError("Not enough data to train model")

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values

    split = int(len(df) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    svr = SVR(kernel="rbf")

    param_grid = {
        "C": [1, 10, 100],
        "gamma": [0.01, 0.1, 1],
        "epsilon": [0.01, 0.1, 1]
    }

    grid = GridSearchCV(
        svr,
        param_grid,
        scoring="neg_mean_squared_error",
        cv=3
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return best_model, mse, mae


def forecast_next_days(model, df, days=7):
    df = df.copy()
    df = df.dropna()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.tail(60)

    X_all = np.arange(len(df)).reshape(-1, 1)
    last_x = X_all[-1][0]

    X_future = np.arange(last_x + 1, last_x + 1 + days).reshape(-1, 1)

    return model.predict(X_future)
