"""
Consumption DNA - Section 8.

Derives an interpretable behavioural profile purely from measurable
statistics on the active dataset. Every trait label comes from a
DOCUMENTED, fixed numeric threshold applied to a real calculated value -
never an arbitrary or invented judgement. The thresholds themselves are
returned in the response so the "why" is always inspectable, not a black
box.
"""
import pandas as pd
import numpy as np
from .anomaly import detect_anomalies

NIGHT_HOURS = set(range(23, 24)) | set(range(0, 6))   # 23:00-05:59
EVENING_HOURS = set(range(17, 21))                     # 17:00-20:59

THRESHOLDS = {
    "peak_concentration_pct": {"high": 12.5, "medium": 6.25},   # fair share of 1 hour = 100/24 = 4.17%
    "night_share_pct": {"high": 35, "medium": 20},               # fair share of 6 night hours = 25%
    "coefficient_of_variation": {"high": 0.35, "medium": 0.18},
    "weekend_deviation_pct": {"high": 20, "medium": 8},
    "anomaly_rate_pct": {"high": 8, "medium": 3},
}


def _level(value, high, medium):
    if value >= high:
        return "HIGH"
    if value >= medium:
        return "MEDIUM"
    return "LOW"


def compute_consumption_dna(hourly: pd.DataFrame, daily: pd.DataFrame):
    if len(daily) < 7:
        return {
            "available": False,
            "message": "Need at least 7 days of data to compute a reliable Consumption DNA profile.",
        }

    df = hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    total_kwh = float(df["consumption_kwh"].sum()) or 1.0
    by_hour = df.groupby("hour")["consumption_kwh"].sum()
    peak_hour = int(by_hour.idxmax())
    peak_hour_share_pct = round(float(by_hour.max()) / total_kwh * 100, 2)

    night_kwh = float(df.loc[df["hour"].isin(NIGHT_HOURS), "consumption_kwh"].sum())
    night_share_pct = round(night_kwh / total_kwh * 100, 2)

    evening_kwh = float(df.loc[df["hour"].isin(EVENING_HOURS), "consumption_kwh"].sum())
    evening_share_pct = round(evening_kwh / total_kwh * 100, 2)

    mean_daily = float(daily["consumption_kwh"].mean())
    std_daily = float(daily["consumption_kwh"].std(ddof=0))
    coefficient_of_variation = round(std_daily / mean_daily, 3) if mean_daily else 0.0

    weekend_avg = float(df.loc[df["is_weekend"], "consumption_kwh"].mean() or 0)
    weekday_avg = float(df.loc[~df["is_weekend"], "consumption_kwh"].mean() or 0)
    weekend_deviation_pct = round((weekend_avg - weekday_avg) / weekday_avg * 100, 2) if weekday_avg else 0.0

    anomaly_rate_pct = 0.0
    anomalies_count = 0
    try:
        anom = detect_anomalies(daily)
        anomalies_count = anom["anomalies_found"]
        anomaly_rate_pct = round(anomalies_count / len(daily) * 100, 2)
    except ValueError:
        pass  # not enough data for anomaly detection - rate stays 0, not fabricated

    traits = {
        "peak_behaviour": {
            "value_pct": peak_hour_share_pct, "peak_hour": peak_hour,
            "level": _level(peak_hour_share_pct, THRESHOLDS["peak_concentration_pct"]["high"], THRESHOLDS["peak_concentration_pct"]["medium"]),
        },
        "night_usage": {
            "value_pct": night_share_pct,
            "level": _level(night_share_pct, THRESHOLDS["night_share_pct"]["high"], THRESHOLDS["night_share_pct"]["medium"]),
        },
        "variability": {
            "value": coefficient_of_variation,
            "level": _level(coefficient_of_variation, THRESHOLDS["coefficient_of_variation"]["high"], THRESHOLDS["coefficient_of_variation"]["medium"]),
        },
        "weekend_pattern": {
            "value_pct": weekend_deviation_pct,
            "direction": "weekend-heavy" if weekend_deviation_pct > 0 else "weekday-heavy" if weekend_deviation_pct < 0 else "balanced",
            "level": _level(abs(weekend_deviation_pct), THRESHOLDS["weekend_deviation_pct"]["high"], THRESHOLDS["weekend_deviation_pct"]["medium"]),
        },
        "anomaly_frequency": {
            "value_pct": anomaly_rate_pct, "count": anomalies_count,
            "level": _level(anomaly_rate_pct, THRESHOLDS["anomaly_rate_pct"]["high"], THRESHOLDS["anomaly_rate_pct"]["medium"]),
        },
    }

    # Deterministic profile assignment - first matching rule wins, documented order
    if traits["peak_behaviour"]["level"] == "HIGH" and evening_share_pct >= night_share_pct:
        profile = "Evening-Peak Consumer"
        reason = f"{peak_hour_share_pct}% of consumption concentrates in a single hour ({peak_hour}:00), with evening (17:00-21:00) as the dominant segment."
    elif traits["night_usage"]["level"] == "HIGH":
        profile = "Night-Heavy Consumer"
        reason = f"{night_share_pct}% of consumption occurs overnight (23:00-06:00), well above a proportional 25% share."
    elif traits["weekend_pattern"]["level"] == "HIGH" and traits["weekend_pattern"]["direction"] == "weekend-heavy":
        profile = "Weekend-Heavy Consumer"
        reason = f"Weekend average is {weekend_deviation_pct}% higher than weekday average."
    elif traits["variability"]["level"] == "HIGH":
        profile = "Irregular / Volatile Consumer"
        reason = f"Day-to-day variability (coefficient of variation {coefficient_of_variation}) is high relative to the mean."
    elif all(t["level"] == "LOW" for t in traits.values()):
        profile = "Stable / Balanced Consumer"
        reason = "All measured traits fall within low/typical ranges - no single behaviour dominates."
    else:
        profile = "Mixed-Pattern Consumer"
        reason = "No single trait dominates strongly enough to assign a more specific profile."

    return {
        "available": True,
        "profile": profile,
        "profile_reason": reason,
        "traits": traits,
        "thresholds": THRESHOLDS,
        "methodology": (
            "Each trait is computed directly from the active dataset (peak-hour share of total kWh, "
            "night-hour share, day-to-day coefficient of variation, weekend-vs-weekday deviation, and "
            "anomaly rate from the existing anomaly-detection module). Levels (LOW/MEDIUM/HIGH) come from "
            "the fixed numeric thresholds shown above, not subjective judgement. The profile label is "
            "assigned by a documented, deterministic rule order - not a machine-learned classification."
        ),
    }
