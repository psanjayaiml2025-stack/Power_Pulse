"""
Analytics for periodic bill/meter-reading data.

Deliberately separate from the hourly smart-meter pipeline (data_pipeline.py
/ forecasting.py / anomaly.py) because bill data has fundamentally
different structure: few, irregularly-spaced periods (monthly/bi-monthly)
rather than dense timestamps. Every figure here is derived only from
aggregate units-consumed per period - never appliance-level, since a
standard EB meter/bill cannot support that.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from .anomaly import detect_anomalies
from .tariffs import estimate_cost


def eda_summary(series: pd.DataFrame):
    if series.empty:
        return {"available": False, "message": "No consumption periods yet - add at least two readings."}

    stats = series["daily_rate"].describe()
    trend = [{"date": r["date"], "units_consumed": r["units_consumed"], "daily_rate": r["daily_rate"],
              "period_days": r["period_days"]} for _, r in series.iterrows()]

    monthly = series.copy()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.month
    seasonal = [
        {"month": int(m), "avg_daily_rate": round(float(v), 3)}
        for m, v in monthly.groupby("month")["daily_rate"].mean().items()
    ]

    return {
        "available": True,
        "periods_count": int(len(series)),
        "descriptive_stats": {k: round(float(v), 4) for k, v in stats.to_dict().items()},
        "trend": trend,
        "seasonal_by_month": seasonal,
    }


def detect_meter_anomalies(series: pd.DataFrame):
    if len(series) < 6:
        return {
            "available": False,
            "message": f"Need at least 6 reading periods for reliable anomaly detection (have {len(series)}).",
            "anomalies": [],
        }
    adapted = series.rename(columns={"daily_rate": "consumption_kwh"})[["date", "consumption_kwh"]]
    window = min(3, max(3, len(adapted) // 3))
    try:
        result = detect_anomalies(adapted, window=window, z_thresh=2.0)
    except ValueError as e:
        return {"available": False, "message": str(e), "anomalies": []}

    # translate back to period-consumption language
    for a in result["anomalies"]:
        a["actual_daily_rate_kwh"] = a.pop("actual_kwh")
        a["expected_daily_rate_low"] = a.pop("expected_range_low")
        a["expected_daily_rate_high"] = a.pop("expected_range_high")

    return {"available": True, "method": result["method"], "anomalies_found": result["anomalies_found"], "anomalies": result["anomalies"]}


def forecast_next_period(series: pd.DataFrame, next_period_days: int = 30):
    if len(series) < 4:
        return {"available": False, "message": f"Need at least 4 reading periods to forecast (have {len(series)})."}

    y = series["daily_rate"].values
    x = np.arange(len(y)).reshape(-1, 1)

    naive_pred = float(np.mean(y[-3:]))

    model_used = "3-period moving average (insufficient data for trend regression)"
    forecast_rate = naive_pred
    evaluation = None

    if len(series) >= 6:
        # Backtest: leave last point out
        model = LinearRegression().fit(x[:-1], y[:-1])
        backtest_pred = float(model.predict(x[-1:])[0])
        lr_mae = abs(backtest_pred - y[-1])
        naive_backtest = float(np.mean(y[-4:-1]))
        naive_mae = abs(naive_backtest - y[-1])

        model_full = LinearRegression().fit(x, y)
        lr_forecast = float(model_full.predict(np.array([[len(y)]]))[0])

        if lr_mae <= naive_mae:
            forecast_rate = lr_forecast
            model_used = "Linear trend regression (outperformed naive average on backtest)"
        else:
            model_used = "3-period moving average (outperformed linear trend on backtest)"
        evaluation = {"linear_regression_MAE": round(lr_mae, 3), "moving_avg_MAE": round(naive_mae, 3)}

    forecast_rate = max(0.0, forecast_rate)
    projected_units = round(forecast_rate * next_period_days, 2)

    return {
        "available": True,
        "model_used": model_used,
        "evaluation": evaluation,
        "forecast_daily_rate": round(forecast_rate, 3),
        "next_period_days_assumed": next_period_days,
        "projected_units_next_period": projected_units,
        "message": f"Based on your recent reading pattern, projected usage for the next "
                   f"{next_period_days}-day period is about {projected_units} units.",
    }


def whatif_meter(series: pd.DataFrame, reduction_pct: float, state: str, custom_rate, next_period_days: int = 30):
    if series.empty:
        return {"available": False, "message": "No reading history yet."}

    avg_daily_rate = float(series["daily_rate"].mean())
    current_projection = avg_daily_rate * next_period_days
    scenario_daily_rate = avg_daily_rate * (1 - reduction_pct / 100.0)
    scenario_projection = scenario_daily_rate * next_period_days

    current_cost = estimate_cost(current_projection, state, custom_rate)
    scenario_cost = estimate_cost(scenario_projection, state, custom_rate)

    return {
        "available": True,
        "current_avg_daily_rate": round(avg_daily_rate, 3),
        "scenario_avg_daily_rate": round(scenario_daily_rate, 3),
        "current_projected_units": round(current_projection, 2),
        "scenario_projected_units": round(scenario_projection, 2),
        "units_saved": round(current_projection - scenario_projection, 2),
        "current_projected_cost": current_cost["estimated_cost"],
        "scenario_projected_cost": scenario_cost["estimated_cost"],
        "estimated_saving": round(current_cost["estimated_cost"] - scenario_cost["estimated_cost"], 2),
        "rate_basis": current_cost["rate_basis"],
    }


def generate_meter_recommendations(series: pd.DataFrame, anomaly_result: dict):
    recs = []
    if series.empty:
        return [{
            "observation": "No reading history yet.",
            "evidence": "Add at least two meter readings or bills to start generating insights.",
            "suggested_action": "Enter your most recent bill's units-consumed figure, or a manual meter reading.",
            "expected_impact": "Unlocks trend, anomaly, and forecast analysis.",
            "severity": "low",
        }]

    if len(series) >= 4:
        recent = series["daily_rate"].tail(2).mean()
        prior = series["daily_rate"].iloc[:-2].mean() if len(series) > 2 else series["daily_rate"].mean()
        if prior and (recent - prior) / prior > 0.15:
            recs.append({
                "observation": "Your recent daily consumption rate is trending upward.",
                "evidence": f"Last 2 periods avg {recent:.2f} units/day vs. earlier average {prior:.2f} units/day "
                            f"(+{(recent/prior-1)*100:.0f}%).",
                "suggested_action": "Check for new appliances, more AC usage, or occupancy changes in recent months.",
                "expected_impact": "Identifying the cause could plateau or reverse the trend.",
                "severity": "high",
            })

    if anomaly_result.get("available") and anomaly_result["anomalies_found"] > 0:
        a = anomaly_result["anomalies"][-1]
        recs.append({
            "observation": f"{anomaly_result['anomalies_found']} unusual billing period(s) detected.",
            "evidence": f"Period ending {a['date']}: {a['actual_daily_rate_kwh']} units/day vs. expected "
                        f"{a['expected_daily_rate_low']}-{a['expected_daily_rate_high']} units/day.",
            "suggested_action": "Cross-check that period's bill for meter errors, tariff changes, or unusual usage.",
            "expected_impact": "Confirms whether the anomaly is a real usage change or a billing/meter issue.",
            "severity": "high",
        })

    gaps = pd.to_datetime(series["date"]).diff().dt.days.dropna()
    if len(gaps) and gaps.max() > 70:
        recs.append({
            "observation": "There's a large gap between some of your readings.",
            "evidence": f"Longest gap between consecutive readings: {int(gaps.max())} days.",
            "suggested_action": "Take readings at more regular intervals (e.g. every bill cycle) for more reliable trend and forecast analysis.",
            "expected_impact": "More consistent, shorter gaps improve forecast and anomaly accuracy.",
            "severity": "moderate",
        })

    if not recs:
        recs.append({
            "observation": "No significant irregularities in your billing history.",
            "evidence": "Trend and anomaly checks both fall within normal ranges.",
            "suggested_action": "Continue current usage patterns; keep adding readings each cycle.",
            "expected_impact": "Maintains current efficiency and analysis accuracy.",
            "severity": "low",
        })
    return recs


def build_meter_dashboard(series: pd.DataFrame, state: str, custom_rate, benchmark_daily_rate=None):
    if series.empty:
        return {
            "available": False,
            "message": "Add at least two meter readings or bill entries to generate your dashboard.",
        }

    latest = series.iloc[-1]
    historical_avg = float(series["daily_rate"].iloc[:-1].mean()) if len(series) > 1 else float(series["daily_rate"].mean())
    pct_change = round((latest["daily_rate"] - historical_avg) / historical_avg * 100, 1) if historical_avg else None

    anomaly_result = detect_meter_anomalies(series)
    forecast_result = forecast_next_period(series)
    recs = generate_meter_recommendations(series, anomaly_result)

    monthly_projection = round(float(series["daily_rate"].mean()) * 30, 2)
    cost = estimate_cost(monthly_projection, state, custom_rate)

    if anomaly_result.get("available") and anomaly_result["anomalies_found"] > 0:
        top_insight = f"{anomaly_result['anomalies_found']} unusual billing period(s) detected in your history."
    elif pct_change is not None and abs(pct_change) > 10:
        direction = "higher" if pct_change > 0 else "lower"
        top_insight = f"Your latest period's daily usage is {abs(pct_change)}% {direction} than your own historical average."
    else:
        top_insight = "Your latest reading is within your normal historical range."

    benchmark = None
    if benchmark_daily_rate:
        benchmark = {
            "reference_daily_rate": round(benchmark_daily_rate, 3),
            "note": "For reference only - a general household benchmark from a different household/region, illustrative.",
        }

    return {
        "available": True,
        "latest_reading_date": latest["date"],
        "latest_daily_rate": latest["daily_rate"],
        "latest_units_consumed": latest["units_consumed"],
        "historical_avg_daily_rate": round(historical_avg, 3),
        "pct_change_vs_own_history": pct_change,
        "monthly_projected_units": monthly_projection,
        "monthly_projected_cost": cost,
        "anomalies_count": anomaly_result.get("anomalies_found", 0) if anomaly_result.get("available") else 0,
        "forecast": forecast_result,
        "benchmark": benchmark,
        "top_insight": top_insight,
        "recommendations_preview": recs[:2],
    }
