"""
Anomaly detection module.

Two methods are computed and compared, per Section 10:
  1. Rolling mean/std z-score (interpretable statistical baseline)
  2. Isolation Forest (learns a more general notion of "unusual")

Both are run; points flagged by BOTH are marked high-severity, points
flagged by only one are marked moderate. Nothing is invented - anomalies
are only reported where the underlying data actually deviates.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(daily: pd.DataFrame, window: int = 7, z_thresh: float = 2.5):
    df = daily.copy().sort_values("date").reset_index(drop=True)
    if len(df) < window + 3:
        raise ValueError(f"Need at least {window + 3} days of data for anomaly detection.")

    df["rolling_mean"] = df["consumption_kwh"].rolling(window, min_periods=3).mean()
    df["rolling_std"] = df["consumption_kwh"].rolling(window, min_periods=3).std().replace(0, np.nan)
    df["zscore"] = (df["consumption_kwh"] - df["rolling_mean"]) / df["rolling_std"]
    df["z_flag"] = df["zscore"].abs() >= z_thresh

    iso = IsolationForest(contamination=0.08, random_state=42, n_estimators=200)
    feat = df[["consumption_kwh"]].fillna(df["consumption_kwh"].median())
    df["iso_flag"] = iso.fit_predict(feat) == -1

    df["expected_low"] = (df["rolling_mean"] - z_thresh * df["rolling_std"]).clip(lower=0)
    df["expected_high"] = df["rolling_mean"] + z_thresh * df["rolling_std"]

    anomalies = []
    for _, row in df.iterrows():
        if not (row["z_flag"] or row["iso_flag"]):
            continue
        if pd.isna(row["rolling_mean"]):
            continue
        severity = "high" if (row["z_flag"] and row["iso_flag"]) else "moderate"
        deviation_pct = (
            ((row["consumption_kwh"] - row["rolling_mean"]) / row["rolling_mean"]) * 100
            if row["rolling_mean"] else 0
        )
        anomalies.append({
            "date": str(row["date"]),
            "actual_kwh": round(float(row["consumption_kwh"]), 3),
            "expected_range_low": round(float(row["expected_low"]), 3) if not pd.isna(row["expected_low"]) else None,
            "expected_range_high": round(float(row["expected_high"]), 3) if not pd.isna(row["expected_high"]) else None,
            "deviation_pct": round(float(deviation_pct), 1),
            "severity": severity,
            "flagged_by": (
                "statistical (z-score) + Isolation Forest" if severity == "high"
                else ("z-score" if row["z_flag"] else "Isolation Forest")
            ),
        })

    return {
        "method": "Rolling z-score (window=%d, threshold=%.1f) + Isolation Forest, cross-validated against each other" % (window, z_thresh),
        "total_points": int(len(df)),
        "anomalies_found": len(anomalies),
        "anomalies": anomalies,
    }
