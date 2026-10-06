"""
Dashboard intelligence layer - upgrade per FINAL UPGRADE INSTRUCTION.

Builds the full "Energy Intelligence Dashboard" bundle in one call:
current-vs-baseline comparison, where-is-my-energy-going, why-did-it-
increase, and if-I-continue-like-this future impact. Every number comes
from actual computed data - nothing is hard-coded.
"""
import pandas as pd
import numpy as np
from .tariffs import estimate_cost
from .forecasting import train_and_forecast
from .anomaly import detect_anomalies
from .health_score import compute_health_score
from .whatif import run_whatif

SEGMENTS = [
    ("Night (00:00-06:00)", 0, 6),
    ("Morning (06:00-12:00)", 6, 12),
    ("Afternoon (12:00-17:00)", 12, 17),
    ("Evening (17:00-21:00)", 17, 21),
    ("Late Night (21:00-24:00)", 21, 24),
]


def _hour_of_day_avg(hourly: pd.DataFrame):
    df = hourly.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    return df.groupby("hour")["consumption_kwh"].mean()


def _segment_avgs(hour_avg: pd.Series):
    out = {}
    for label, start, end in SEGMENTS:
        hrs = [h for h in range(start, end) if h in hour_avg.index]
        out[label] = float(hour_avg.loc[hrs].mean()) if hrs else 0.0
    return out


def build_dashboard(current_hourly, current_daily, baseline_hourly, baseline_daily,
                     has_current: bool, state: str = "Tamil Nadu", custom_rate=None):
    avg_daily_current = float(current_daily["consumption_kwh"].mean()) if len(current_daily) else 0.0
    avg_daily_baseline = float(baseline_daily["consumption_kwh"].mean()) if len(baseline_daily) else 0.0
    today_kwh = float(current_daily.sort_values("date").iloc[-1]["consumption_kwh"]) if len(current_daily) else 0.0

    pct_change_vs_baseline = None
    if avg_daily_baseline > 0:
        pct_change_vs_baseline = round((avg_daily_current - avg_daily_baseline) / avg_daily_baseline * 100, 1)

    cur_hour_avg = _hour_of_day_avg(current_hourly)
    cur_segments = _segment_avgs(cur_hour_avg)
    peak_period_label = max(cur_segments, key=cur_segments.get) if cur_segments else None
    total_avg = sum(cur_segments.values()) or 1
    peak_period_share_pct = round(cur_segments.get(peak_period_label, 0) / total_avg * 100, 1) if peak_period_label else None
    peak_hour = int(cur_hour_avg.idxmax()) if len(cur_hour_avg) else None

    dfc = current_hourly.copy()
    dfc["is_weekend"] = pd.to_datetime(dfc["timestamp"]).dt.dayofweek.isin([5, 6])
    wk = dfc.groupby("is_weekend")["consumption_kwh"].mean()
    weekday_vs_weekend = {
        "weekday_avg_kwh": round(float(wk.get(False, 0)), 3),
        "weekend_avg_kwh": round(float(wk.get(True, 0)), 3),
    }

    monthly_projected_kwh = round(avg_daily_current * 30, 2)
    cost = estimate_cost(monthly_projected_kwh, state, custom_rate)

    # Anomalies
    anomalies_count = 0
    try:
        anom_result = detect_anomalies(current_daily)
        anomalies_count = anom_result["anomalies_found"]
    except ValueError:
        anom_result = None

    # WHY DID IT INCREASE - compare current vs baseline by time segment
    why_increased = None
    if has_current and len(baseline_hourly):
        base_hour_avg = _hour_of_day_avg(baseline_hourly)
        base_segments = _segment_avgs(base_hour_avg)
        best_label, best_pct = None, 0
        for label in cur_segments:
            b = base_segments.get(label, 0)
            if b > 0:
                delta_pct = (cur_segments[label] - b) / b * 100
                if delta_pct > best_pct:
                    best_pct = delta_pct
                    best_label = label
        if best_label and best_pct > 10:
            why_increased = {
                "segment": best_label,
                "pct_increase": round(best_pct, 1),
                "message": f"Your {best_label.split(' (')[0].lower()} consumption is "
                           f"{round(best_pct)}% higher than your historical baseline, "
                           f"with the largest increase during {best_label.split('(')[1].rstrip(')')}.",
            }

    # IF I CONTINUE LIKE THIS - future impact
    future_impact = None
    try:
        fc = train_and_forecast(current_daily, horizon_days=7)
        next7_kwh = round(sum(f["forecast_kwh"] for f in fc["forecast"]), 2)
        projected_monthly = round(next7_kwh / 7 * 30, 2)
        vs_baseline_monthly_pct = None
        if avg_daily_baseline > 0:
            vs_baseline_monthly_pct = round((projected_monthly - avg_daily_baseline * 30) / (avg_daily_baseline * 30) * 100, 1)
        future_cost = estimate_cost(projected_monthly, state, custom_rate)
        future_impact = {
            "next_7_day_kwh": next7_kwh,
            "projected_monthly_kwh": projected_monthly,
            "projected_monthly_cost": future_cost["estimated_cost"],
            "vs_baseline_monthly_pct": vs_baseline_monthly_pct,
            "model_used": fc["model_used"],
            "message": f"Based on your recent consumption pattern, you're projected to use about "
                       f"{projected_monthly} kWh next month" +
                       (f" ({'+' if vs_baseline_monthly_pct and vs_baseline_monthly_pct > 0 else ''}"
                        f"{vs_baseline_monthly_pct}% vs. your historical baseline)." if vs_baseline_monthly_pct is not None else "."),
        }
    except ValueError:
        future_impact = None

    # Top insight headline
    if anomalies_count > 0:
        top_insight = f"{anomalies_count} unusual consumption day(s) detected in your recent data."
    elif why_increased:
        top_insight = why_increased["message"]
    elif pct_change_vs_baseline is not None and abs(pct_change_vs_baseline) > 10:
        direction = "higher" if pct_change_vs_baseline > 0 else "lower"
        top_insight = f"Your average usage is {abs(pct_change_vs_baseline)}% {direction} than the historical baseline."
    else:
        top_insight = "Your consumption is within normal historical patterns."

    health = compute_health_score(
        pct_change_vs_baseline, anomalies_count, len(current_daily),
        peak_period_share_pct,
        future_impact["vs_baseline_monthly_pct"] if future_impact else None,
    )

    # Potential Savings KPI - Section 2. A default modest scenario (10%
    # peak-hour reduction), clearly labeled as a simulation, not a promise.
    try:
        potential_savings = run_whatif(current_hourly, 10.0, 0.0, 0.0, state, custom_rate)
        potential_savings["scenario_label"] = "Simulation: 10% peak-hour reduction — not a guaranteed saving."
    except Exception:
        potential_savings = None

    # Executive insight (Section 2) - one-paragraph synthesis of the above
    exec_parts = []
    if pct_change_vs_baseline is not None:
        direction = "above" if pct_change_vs_baseline > 0 else "below"
        exec_parts.append(f"Consumption is {abs(pct_change_vs_baseline)}% {direction} the learned baseline.")
    if why_increased:
        exec_parts.append(f"The largest increase occurred during {why_increased['segment'].split(' (')[0].lower()} periods.")
    if anomalies_count > 0:
        exec_parts.append(f"{anomalies_count} day(s) show unusual consumption patterns.")
    if not exec_parts:
        exec_parts.append("Consumption is tracking within normal historical patterns.")
    executive_insight = " ".join(exec_parts)

    return {
        "has_current_data": has_current,
        "today_kwh": round(today_kwh, 3),
        "avg_daily_kwh": round(avg_daily_current, 3),
        "baseline_avg_daily_kwh": round(avg_daily_baseline, 3),
        "pct_change_vs_baseline": pct_change_vs_baseline,
        "peak_hour": peak_hour,
        "peak_period_label": peak_period_label,
        "peak_period_share_pct": peak_period_share_pct,
        "segment_breakdown": {k: round(v, 3) for k, v in cur_segments.items()},
        "weekday_vs_weekend": weekday_vs_weekend,
        "monthly_projected_kwh": monthly_projected_kwh,
        "monthly_projected_cost": cost,
        "anomalies_count": anomalies_count,
        "why_increased": why_increased,
        "future_impact": future_impact,
        "top_insight": top_insight,
        "executive_insight": executive_insight,
        "health_score": health,
        "potential_savings": potential_savings,
    }
