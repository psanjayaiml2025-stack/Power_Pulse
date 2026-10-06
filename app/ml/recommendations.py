"""Rule-based recommendation engine driven by actual analysis results
(EDA + anomaly + forecast outputs) - not generic advice, per Section 14."""
import pandas as pd


def generate_recommendations(hourly: pd.DataFrame, daily: pd.DataFrame, anomalies: list):
    recs = []
    df = hourly.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

    by_hour = df.groupby("hour")["consumption_kwh"].mean()
    if len(by_hour):
        peak_hour = int(by_hour.idxmax())
        peak_val = float(by_hour.max())
        avg_val = float(by_hour.mean())
        if peak_val > avg_val * 1.3:
            recs.append({
                "observation": f"Consumption consistently peaks around {peak_hour}:00.",
                "evidence": f"Average usage at {peak_hour}:00 is {peak_val:.2f} kWh vs an all-hour average of {avg_val:.2f} kWh (+{(peak_val/avg_val-1)*100:.0f}%).",
                "suggested_action": "Shift flexible/high-draw appliance use (washing machine, water heater) away from this hour where possible.",
                "expected_impact": "Lower peak-period draw and potential tariff savings if time-of-day pricing applies.",
                "severity": "moderate",
            })

    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["dow"].isin([5, 6])
    wk = df.groupby("is_weekend")["consumption_kwh"].mean()
    if True in wk.index and False in wk.index:
        weekend_avg, weekday_avg = wk.get(True, 0), wk.get(False, 0)
        if weekday_avg and abs(weekend_avg - weekday_avg) / weekday_avg > 0.2:
            direction = "higher" if weekend_avg > weekday_avg else "lower"
            recs.append({
                "observation": f"Weekend consumption is notably {direction} than weekday consumption.",
                "evidence": f"Weekend avg: {weekend_avg:.2f} kWh/hr, weekday avg: {weekday_avg:.2f} kWh/hr.",
                "suggested_action": "Review weekend appliance usage patterns" if direction == "higher" else "No action needed - weekend usage is already efficient.",
                "expected_impact": "Better matching of usage to actual need.",
                "severity": "low",
            })

    if len(daily) >= 14:
        d = daily.sort_values("date")
        first_half = d.iloc[: len(d) // 2]["consumption_kwh"].mean()
        second_half = d.iloc[len(d) // 2 :]["consumption_kwh"].mean()
        if first_half and (second_half - first_half) / first_half > 0.15:
            recs.append({
                "observation": "Daily consumption shows a rising trend over the analyzed period.",
                "evidence": f"First-half average: {first_half:.2f} kWh/day, second-half average: {second_half:.2f} kWh/day (+{(second_half/first_half-1)*100:.0f}%).",
                "suggested_action": "Investigate new appliances, occupancy changes, or seasonal effects (e.g. AC use) driving the increase.",
                "expected_impact": "Identifying the driver could reverse or plateau the trend.",
                "severity": "high",
            })

    high_sev_anoms = [a for a in anomalies if a.get("severity") == "high"]
    if high_sev_anoms:
        recs.append({
            "observation": f"{len(high_sev_anoms)} high-severity consumption anomaly day(s) detected.",
            "evidence": f"Most recent: {high_sev_anoms[-1]['date']}, actual {high_sev_anoms[-1]['actual_kwh']} kWh vs expected range "
                        f"{high_sev_anoms[-1]['expected_range_low']}-{high_sev_anoms[-1]['expected_range_high']} kWh.",
            "suggested_action": "Check for appliances left running, faulty equipment, or unusual occupancy on flagged dates.",
            "expected_impact": "Fixing the root cause could recover the excess consumption.",
            "severity": "high",
        })

    if not recs:
        recs.append({
            "observation": "No significant irregular patterns detected in the analyzed period.",
            "evidence": "Peak-hour, weekday/weekend, trend, and anomaly checks all fell within normal ranges.",
            "suggested_action": "Continue current usage patterns; re-check periodically as new data comes in.",
            "expected_impact": "Maintains current efficiency.",
            "severity": "low",
        })

    return recs
