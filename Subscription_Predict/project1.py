# ==========================================================
# ADVANCED SUBSCRIBER PREDICTION SYSTEM
# Levels 1–4 Combined (ML + DL + Business + Production)
# ==========================================================

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------------------------
# 1. DATA GENERATION (NO EXTERNAL DATASET REQUIRED)
# ----------------------------------------------------------
def generate_data():
    np.random.seed(42)
    months = pd.date_range(start="2020-01-01", periods=60, freq="M")
    subscribers = np.cumsum(np.random.randint(40, 90, size=len(months))) + 1000
    return pd.DataFrame({"Month": months, "Subscribers": subscribers})

# ----------------------------------------------------------
# 2. FEATURE ENGINEERING (LEVEL 1)
# ----------------------------------------------------------
def add_features(df):
    df = df.copy()
    df["lag_1"] = df["Subscribers"].shift(1)
    df["lag_3"] = df["Subscribers"].shift(3)
    df["rolling_mean_3"] = df["Subscribers"].rolling(3).mean()
    df.dropna(inplace=True)
    return df

# ----------------------------------------------------------
# 3. ROLLING WINDOW VALIDATION (LEVEL 1)
# ----------------------------------------------------------
def rolling_validation(df, window=24, horizon=3):
    maes = []
    for i in range(len(df) - window - horizon):
        train = df.iloc[i:i + window]
        test = df.iloc[i + window:i + window + horizon]

        X_train = train.drop(["Subscribers", "Month"], axis=1)
        y_train = train["Subscribers"]
        X_test = test.drop(["Subscribers", "Month"], axis=1)
        y_test = test["Subscribers"]

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))

    return np.mean(maes)

# ----------------------------------------------------------
# 4. TRAIN XGBOOST MODEL (LEVEL 2)
# ----------------------------------------------------------
def train_xgboost(df):
    train = df.iloc[:-6]
    test = df.iloc[-6:]

    X_train = train.drop(["Subscribers", "Month"], axis=1)
    y_train = train["Subscribers"]
    X_test = test.drop(["Subscribers", "Month"], axis=1)
    y_test = test["Subscribers"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    joblib.dump(model, f"{MODEL_DIR}/xgboost_model.pkl")

    return model, mae, preds, test

# ----------------------------------------------------------
# 5. FEATURE IMPORTANCE (LEVEL 2)
# ----------------------------------------------------------
def plot_feature_importance(model, features):
    plt.figure(figsize=(6, 4))
    plt.barh(features, model.feature_importances_)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------
# 6. BUSINESS METRICS & SCENARIOS (LEVEL 2)
# ----------------------------------------------------------
def business_metrics(current, prediction):
    growth = ((prediction - current) / current) * 100
    churn = prediction * 0.9
    marketing = prediction * 1.15

    print("\nBusiness Metrics")
    print("----------------")
    print(f"Current Subscribers : {int(current)}")
    print(f"Predicted Next Month: {int(prediction)}")
    print(f"Growth Rate (%)     : {growth:.2f}%")

    print("\nScenario Analysis")
    print("-----------------")
    print("High Churn Scenario      :", int(churn))
    print("Marketing Boost Scenario :", int(marketing))

# ----------------------------------------------------------
# 7. LSTM MODEL (LEVEL 3)
# ----------------------------------------------------------
def train_lstm(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Subscribers"]])

    X, y = [], []
    window = 6
    for i in range(len(scaled) - window):
        X.append(scaled[i:i + window])
        y.append(scaled[i + window])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, input_shape=(window, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    preds = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test, preds)
    model.save(f"{MODEL_DIR}/lstm_model.keras")

    return mae

# ----------------------------------------------------------
# 8. MAIN PIPELINE (LEVEL 4)
# ----------------------------------------------------------
def main():
    logging.info("Starting Advanced Subscriber Prediction System")

    df_raw = generate_data()
    df_feat = add_features(df_raw)

    rolling_mae = rolling_validation(df_feat)
    logging.info(f"Rolling Window MAE: {rolling_mae:.2f}")

    xgb_model, xgb_mae, preds, test = train_xgboost(df_feat)
    logging.info(f"XGBoost MAE: {xgb_mae:.2f}")

    lstm_mae = train_lstm(df_raw)
    logging.info(f"LSTM MAE: {lstm_mae:.2f}")

    print("\nModel Comparison")
    print("----------------")
    print(f"XGBoost MAE : {xgb_mae:.2f}")
    print(f"LSTM MAE    : {lstm_mae:.2f}")

    plot_feature_importance(xgb_model, df_feat.drop(["Subscribers", "Month"], axis=1).columns)

    next_prediction = xgb_model.predict(
        df_feat.iloc[-1:].drop(["Subscribers", "Month"], axis=1)
    )[0]

    business_metrics(df_feat.iloc[-1]["Subscribers"], next_prediction)

    plt.figure(figsize=(10, 5))
    plt.plot(df_feat["Month"], df_feat["Subscribers"], label="Actual")
    plt.plot(test["Month"], preds, linestyle="--", label="XGBoost Prediction")
    plt.legend()
    plt.title("Advanced Subscriber Forecasting")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    logging.info("Pipeline Completed Successfully")

# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
