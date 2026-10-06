"""
Building/Meter hierarchy analytics for College/Campus and Institution mode
(Sections 4, 12, 16). Only activates when the active dataset actually has
a 'building' or 'meter_id' column - never fabricates a hierarchy for
plain single-series household data.
"""
import pandas as pd
from .anomaly import detect_anomalies


def _group_col(hourly: pd.DataFrame):
    if "building" in hourly.columns and hourly["building"].notna().any():
        return "building"
    if "meter_id" in hourly.columns and hourly["meter_id"].notna().any():
        return "meter_id"
    return None


def building_breakdown(hourly: pd.DataFrame):
    """Per-building/meter consumption ranking with a priority score for
    'where should you investigate first?' (Section 12)."""
    group_col = _group_col(hourly)
    if group_col is None:
        return {
            "available": False,
            "message": (
                "No 'building' or 'meter_id' column detected in the active dataset - "
                "building/meter-level analysis requires one of those columns. This "
                "view only applies in College/Campus or Institution mode with a "
                "multi-meter dataset."
            ),
            "group_by": None,
            "buildings": [],
        }

    df = hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    total_kwh_all = df["consumption_kwh"].sum() or 1.0
    results = []

    for name, g in df.groupby(group_col):
        daily = g.groupby("date")["consumption_kwh"].sum().reset_index()
        daily.columns = ["date", "consumption_kwh"]
        total_kwh = float(g["consumption_kwh"].sum())
        avg_daily = float(daily["consumption_kwh"].mean()) if len(daily) else 0.0
        contribution_pct = round(total_kwh / total_kwh_all * 100, 1)

        growth_pct = None
        if len(daily) >= 4:
            d = daily.sort_values("date")
            first_half = d.iloc[: len(d) // 2]["consumption_kwh"].mean()
            second_half = d.iloc[len(d) // 2 :]["consumption_kwh"].mean()
            if first_half:
                growth_pct = round((second_half - first_half) / first_half * 100, 1)

        anomalies_count = 0
        if len(daily) >= 10:
            try:
                anomalies_count = detect_anomalies(daily)["anomalies_found"]
            except ValueError:
                pass

        by_hour = g.copy()
        by_hour["hour"] = pd.to_datetime(by_hour["timestamp"]).dt.hour
        peak_hour = int(by_hour.groupby("hour")["consumption_kwh"].mean().idxmax()) if len(by_hour) else None

        results.append({
            "name": str(name),
            "total_kwh": round(total_kwh, 2),
            "avg_daily_kwh": round(avg_daily, 3),
            "contribution_pct": contribution_pct,
            "growth_pct": growth_pct,
            "anomalies_count": anomalies_count,
            "peak_hour": peak_hour,
        })

    # Priority Engine: composite score from contribution, growth, anomalies
    for r in results:
        score = 0.0
        reasons = []
        if r["contribution_pct"] > 25:
            score += 40
            reasons.append(f"high consumption ({r['contribution_pct']}% of total)")
        elif r["contribution_pct"] > 12:
            score += 20
        if r["growth_pct"] is not None and r["growth_pct"] > 15:
            score += 30
            reasons.append(f"increasing trend (+{r['growth_pct']}%)")
        if r["anomalies_count"] > 0:
            score += min(30, r["anomalies_count"] * 10)
            reasons.append(f"{r['anomalies_count']} unusual day(s) detected")

        r["priority_score"] = round(score, 1)
        if score >= 50:
            r["priority"] = "HIGH"
        elif score >= 20:
            r["priority"] = "MEDIUM"
        else:
            r["priority"] = "NORMAL"
        r["priority_reasons"] = reasons if reasons else ["no significant issues detected"]

    results.sort(key=lambda r: -r["priority_score"])
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return {
        "available": True,
        "group_by": group_col,
        "total_buildings": len(results),
        "buildings": results,
    }
