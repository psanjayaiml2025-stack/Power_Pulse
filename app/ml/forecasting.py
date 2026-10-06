"""
Forecasting module.

Compares a naive persistence baseline against a RandomForestRegressor
trained on lag + calendar features, using a chronological (time-ordered)
train/test split to avoid leakage - Section 27.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _build_supervised(daily: pd.DataFrame, n_lags: int = 7):
    df = daily.copy().sort_values("date").reset_index(drop=True)
    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["consumption_kwh"].shift(lag)
    df["rolling_mean_7"] = df["consumption_kwh"].shift(1).rolling(7).mean()
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c.startswith("lag_")] + [
        "dayofweek", "month", "rolling_mean_7"
    ]
    return df, feature_cols


def _mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_and_forecast(daily: pd.DataFrame, horizon_days: int = 7):
    if len(daily) < 21:
        raise ValueError(
            "Need at least 21 days of daily data to train a reliable "
            "forecasting model with lag features."
        )

    supervised, feature_cols = _build_supervised(daily)
    if len(supervised) < 10:
        raise ValueError("Not enough data after feature engineering to train/test split.")

    test_size = max(3, min(14, int(len(supervised) * 0.2)))
    train = supervised.iloc[:-test_size]
    test = supervised.iloc[-test_size:]

    X_train, y_train = train[feature_cols], train["consumption_kwh"]
    X_test, y_test = test[feature_cols], test["consumption_kwh"]

    # Baseline: persistence (predict yesterday's value)
    baseline_pred = test["lag_1"].values
    baseline_eval = {
        "MAE": round(mean_absolute_error(y_test, baseline_pred), 3),
        "RMSE": round(mean_squared_error(y_test, baseline_pred) ** 0.5, 3),
        "MAPE": round(_mape(y_test, baseline_pred), 2),
    }

    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_eval = {
        "MAE": round(mean_absolute_error(y_test, model_pred), 3),
        "RMSE": round(mean_squared_error(y_test, model_pred) ** 0.5, 3),
        "MAPE": round(_mape(y_test, model_pred), 2),
    }

    # Forecast uncertainty (Section 12) - derived from actual backtest
    # residuals, never fabricated. Omitted entirely if there aren't enough
    # held-out points (5+) to estimate a residual spread from.
    residual_std = None
    risk_level = None
    if len(y_test) >= 5:
        residuals = model_pred - y_test.values
        residual_std = float(np.std(residuals))
        mean_y_train = float(y_train.mean())
        if mean_y_train:
            rel_spread = residual_std / mean_y_train
            risk_level = "High" if rel_spread > 0.35 else ("Moderate" if rel_spread > 0.15 else "Low")

    chosen_model = model
    model_used = "RandomForestRegressor (vs. persistence baseline)"
    if model_eval["MAE"] > baseline_eval["MAE"]:
        model_used = "Persistence baseline (outperformed RandomForest on this data)"

    # Refit on ALL available data for the actual future forecast
    full_X, full_y = supervised[feature_cols], supervised["consumption_kwh"]
    final_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    final_model.fit(full_X, full_y)

    history_tail = daily.tail(60)
    last_known = supervised.iloc[-1].copy()
    lag_window = list(supervised[[f"lag_{i}" for i in range(1, 8)]].iloc[-1].values[::-1])
    lag_window.append(supervised.iloc[-1]["consumption_kwh"])

    forecasts = []
    cur_date = pd.to_datetime(daily["date"].max())
    for step in range(horizon_days):
        cur_date = cur_date + pd.Timedelta(days=1)
        recent_lags = lag_window[-7:]
        feat = {
            f"lag_{i+1}": recent_lags[-(i + 1)] for i in range(7)
        }
        feat["dayofweek"] = cur_date.dayofweek
        feat["month"] = cur_date.month
        feat["rolling_mean_7"] = float(np.mean(recent_lags))
        X_next = pd.DataFrame([feat])[feature_cols]
        pred = float(final_model.predict(X_next)[0])
        pred = max(pred, 0.0)
        entry = {"date": cur_date.strftime("%Y-%m-%d"), "forecast_kwh": round(pred, 3)}
        if residual_std is not None:
            margin = 1.28 * residual_std  # ~80% interval, assuming approx-normal residuals from backtest
            entry["expected_low"] = round(max(0.0, pred - margin), 3)
            entry["expected_high"] = round(pred + margin, 3)
        forecasts.append(entry)
        lag_window.append(pred)

    importances = sorted(
        zip(feature_cols, final_model.feature_importances_), key=lambda x: -x[1]
    )
    feature_importance = [
        {"feature": f, "importance": round(float(imp), 4)} for f, imp in importances
    ]

    history = [
        {"date": str(row["date"]), "consumption_kwh": round(float(row["consumption_kwh"]), 3)}
        for _, row in history_tail.iterrows()
    ]

    return {
        "model_used": model_used,
        "horizon_days": horizon_days,
        "evaluation": model_eval,
        "baseline_evaluation": baseline_eval,
        "history": history,
        "forecast": forecasts,
        "feature_importance": feature_importance,
        "risk_level": risk_level,
    }
