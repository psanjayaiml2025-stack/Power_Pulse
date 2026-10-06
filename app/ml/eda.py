"""EDA module - Section 16."""
import pandas as pd
import numpy as np


def build_eda(hourly: pd.DataFrame, daily: pd.DataFrame):
    df = hourly.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    df["is_weekend"] = df["dow"].isin([5, 6])

    stats = df["consumption_kwh"].describe()
    descriptive_stats = {
        "consumption_kwh": {k: round(float(v), 4) for k, v in stats.to_dict().items()}
    }

    hourly_pattern = [
        {"hour": int(h), "avg_kwh": round(float(v), 4)}
        for h, v in df.groupby("hour")["consumption_kwh"].mean().items()
    ]
    daily_pattern = [
        {"dayofweek": int(d), "avg_kwh": round(float(v), 4)}
        for d, v in df.groupby("dow")["consumption_kwh"].mean().items()
    ]
    monthly_pattern = [
        {"month": int(m), "avg_kwh": round(float(v), 4)}
        for m, v in df.groupby("month")["consumption_kwh"].mean().items()
    ]

    wk = df.groupby("is_weekend")["consumption_kwh"].mean()
    weekday_vs_weekend = {
        "weekday_avg_kwh": round(float(wk.get(False, 0)), 4),
        "weekend_avg_kwh": round(float(wk.get(True, 0)), 4),
    }

    # Distribution data for histogram/box-plot on the frontend
    quantiles = df["consumption_kwh"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    distribution = {
        "p5": round(float(quantiles.loc[0.05]), 4), "p25": round(float(quantiles.loc[0.25]), 4),
        "median": round(float(quantiles.loc[0.5]), 4), "p75": round(float(quantiles.loc[0.75]), 4),
        "p95": round(float(quantiles.loc[0.95]), 4),
    }
    hist_counts, hist_edges = np.histogram(df["consumption_kwh"].dropna(), bins=15)
    histogram = [
        {"bin_start": round(float(hist_edges[i]), 3), "bin_end": round(float(hist_edges[i + 1]), 3), "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    # Electrical-field correlations - only meaningful when the uploaded
    # smart-meter dataset actually contains voltage/current/frequency.
    electrical_fields_present = [c for c in ["voltage", "current", "frequency"] if c in df.columns]
    numeric_cols = ["consumption_kwh", "hour", "dow", "month"] + electrical_fields_present
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr()["consumption_kwh"].drop("consumption_kwh", errors="ignore")
    top_correlations = [
        {"feature": k, "correlation": round(float(v), 3)}
        for k, v in corr.sort_values(key=lambda s: -s.abs()).items()
    ]

    # Hour x day-of-week heatmap for fast identification of recurring high-use windows.
    pivot = df.pivot_table(index="dow", columns="hour", values="consumption_kwh", aggfunc="mean", fill_value=0)
    heatmap = []
    for d in range(7):
        heatmap.append({"dayofweek": d, "values": [round(float(pivot.get(h, pd.Series()).get(d, 0)), 4) for h in range(24)]})
    peak_row = df.groupby(["dow", "hour"])['consumption_kwh'].mean()
    if len(peak_row):
        (peak_dow, peak_h), peak_value = peak_row.idxmax(), float(peak_row.max())
        explanation = f"The strongest recurring average usage occurs on {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][int(peak_dow)]} around {int(peak_h):02d}:00 ({peak_value:.3f} kWh). Use this window to investigate demand drivers."
    else:
        explanation = "Not enough data to identify a reliable recurring high-use window."

    return {
        "row_count": int(len(df)),
        "feature_count": int(len(df.columns)),
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "descriptive_stats": descriptive_stats,
        "distribution": distribution,
        "histogram": histogram,
        "hourly_pattern": hourly_pattern,
        "daily_pattern": daily_pattern,
        "monthly_pattern": monthly_pattern,
        "weekday_vs_weekend": weekday_vs_weekend,
        "top_correlations": top_correlations,
        "electrical_fields_present": electrical_fields_present,
        "heatmap": heatmap,
        "explanation": explanation,
    }
