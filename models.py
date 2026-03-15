"""
Milestone 3: Model Development, Tuning & Backtesting
- Load historical demand data
- Train ARIMA, XGBoost, LSTM
- Rolling-origin backtest
- Compare MAE, RMSE, Forecast Bias
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ARIMA
import statsmodels.api as sm

# XGBoost
from xgboost import XGBRegressor

# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------
DATA_PATH = "azure_demand_capacity_24m.csv"  # your CSV from previous milestone
TARGET_COL = "requests"                      # what you want to forecast
DATE_COL = "date"
GROUP_REGION = "eastus"                      # filter to one region
GROUP_TIER = "Standard"                      # and one tier

FORECAST_HORIZON = 7                         # days ahead
TRAIN_RATIO = 0.75                           # initial train split
LAG_WINDOW = 30                              # for ML models

# ---------------------------------------------------------------------
# 2. LOAD & PREP DATA
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# Filter one time series: region + tier
ts = df[(df["region"] == GROUP_REGION) & (df["service_tier"] == GROUP_TIER)].copy()
ts = ts.sort_values(DATE_COL)

# Simple univariate series for ARIMA
series = ts[[DATE_COL, TARGET_COL]].set_index(DATE_COL).asfreq("D").interpolate()

# Feature engineering for ML models
ts_ml = ts.copy()
ts_ml = ts_ml.set_index(DATE_COL).asfreq("D").interpolate()

# Time features
ts_ml["dow"] = ts_ml.index.dayofweek
ts_ml["month"] = ts_ml.index.month
ts_ml["day"] = ts_ml.index.day

# Lag features for target
for lag in range(1, LAG_WINDOW + 1):
    ts_ml[f"{TARGET_COL}_lag_{lag}"] = ts_ml[TARGET_COL].shift(lag)

# Drop initial rows with NaN due to lags
ts_ml = ts_ml.dropna()

# Define features & target for ML
feature_cols = [c for c in ts_ml.columns if c != TARGET_COL]
X_all = ts_ml[feature_cols].values
y_all = ts_ml[TARGET_COL].values
dates_all = ts_ml.index

# Train/test split index for rolling backtest start
n = len(ts_ml)
train_size = int(n * TRAIN_RATIO)

#

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    bias = float(np.mean(y_pred - y_true))  # positive = over-forecast
    return mae, rmse, bias



def rolling_backtest_arima(series, start_index, horizon):
    
    
    y_true_all, y_pred_all = [], []

    dates = series.index
    for start in range(start_index, len(series) - horizon + 1):
        train_end_date = dates[start - 1]
        train_series = series.loc[:train_end_date]

        # ARIMA(p,d,q) simple config; tune if needed
        model = sm.tsa.ARIMA(train_series, order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)

        test_dates = dates[start:start + horizon]
        y_true = series.loc[test_dates].values
        y_pred = forecast.values

        y_true_all.extend(list(y_true))
        y_pred_all.extend(list(y_pred))

    return np.array(y_true_all), np.array(y_pred_all)


def rolling_backtest_xgb(X, y, start_index, horizon):
    y_true_all, y_pred_all = [], []

    for start in range(start_index, len(y) - horizon + 1):
        X_train = X[:start]
        y_train = y[:start]

        X_test = X[start:start + horizon]
        y_test = y[start:start + horizon]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(list(y_test))
        y_pred_all.extend(list(y_pred))

    return np.array(y_true_all), np.array(y_pred_all)


def build_lstm_model(input_dim):
    model = Sequential()
    model.add(LSTM(32, activation="tanh", return_sequences=False,
                   input_shape=(1, input_dim)))
    model.add(Dense(1))
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=0.001),
    )
    return model


def rolling_backtest_lstm(X, y, start_index, horizon, epochs=20, batch_size=32):
    
    y_true_all, y_pred_all = [], []

    for start in range(start_index, len(y) - horizon + 1):
        X_train = X[:start]
        y_train = y[:start]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Reshape to (samples, timesteps, features)
        X_train_lstm = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])

        model = build_lstm_model(X_train_lstm.shape[2])
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        model.fit(
            X_train_lstm,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[es],
        )

        # Recursive forecast
        X_hist = X[start - 1].reshape(1, -1)  # last known feature vector
        y_preds_window = []

        for step in range(horizon):
            X_scaled = scaler.transform(X_hist)
            X_lstm = X_scaled.reshape(1, 1, -1)
            y_hat = model.predict(X_lstm, verbose=0)[0, 0]
            y_preds_window.append(y_hat)

           

        y_true = y[start:start + horizon]
        y_true_all.extend(list(y_true))
        y_pred_all.extend(y_preds_window)

    return np.array(y_true_all), np.array(y_pred_all)



print("=== ARIMA backtest ===")
y_true_arima, y_pred_arima = rolling_backtest_arima(
    series[TARGET_COL], start_index=train_size, horizon=FORECAST_HORIZON
)
mae_a, rmse_a, bias_a = regression_metrics(y_true_arima, y_pred_arima)
print(f"ARIMA -> MAE: {mae_a:.2f}, RMSE: {rmse_a:.2f}, Bias: {bias_a:.2f}")

print("\n=== XGBoost backtest ===")
y_true_xgb, y_pred_xgb = rolling_backtest_xgb(
    X_all, y_all, start_index=train_size, horizon=FORECAST_HORIZON
)
mae_x, rmse_x, bias_x = regression_metrics(y_true_xgb, y_pred_xgb)
print(f"XGBoost -> MAE: {mae_x:.2f}, RMSE: {rmse_x:.2f}, Bias: {bias_x:.2f}")

print("\n=== LSTM backtest ===")
y_true_lstm, y_pred_lstm = rolling_backtest_lstm(
    X_all, y_all, start_index=train_size, horizon=FORECAST_HORIZON,
    epochs=15, batch_size=32
)
mae_l, rmse_l, bias_l = regression_metrics(y_true_lstm, y_pred_lstm)
print(f"LSTM -> MAE: {mae_l:.2f}, RMSE: {rmse_l:.2f}, Bias: {bias_l:.2f}")



summary = pd.DataFrame({
    "model": ["ARIMA", "XGBoost", "LSTM"],
    "MAE": [mae_a, mae_x, mae_l],
    "RMSE": [rmse_a, rmse_x, rmse_l],
    "Bias": [bias_a, bias_x, bias_l],
}).sort_values("RMSE")

print("\n=== Model comparison (lower is better) ===")
print(summary)
summary.to_csv("milestone3_model_comparison.csv", index=False)
print("\nSaved: milestone3_model_comparison.csv")
