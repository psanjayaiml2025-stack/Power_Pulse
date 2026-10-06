"""What-If energy saving simulator - transparent, rule-based scenario math."""
import pandas as pd
from .tariffs import estimate_cost

PEAK_HOURS = set(range(18, 23))  # 6 PM - 11 PM, common Indian residential peak window


def run_whatif(hourly: pd.DataFrame, peak_reduction_pct: float, off_peak_reduction_pct: float,
               hours_reduced_per_day: float, state: str, custom_rate: float | None):
    df = hourly.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["is_peak"] = df["hour"].isin(PEAK_HOURS)

    n_days = max(1, (pd.to_datetime(df["timestamp"]).max() - pd.to_datetime(df["timestamp"]).min()).days + 1)
    current_total = df["consumption_kwh"].sum()
    current_daily = current_total / n_days

    peak_kwh = df.loc[df["is_peak"], "consumption_kwh"].sum()
    off_peak_kwh = df.loc[~df["is_peak"], "consumption_kwh"].sum()

    peak_saved = peak_kwh * (peak_reduction_pct / 100.0)
    off_peak_saved = off_peak_kwh * (off_peak_reduction_pct / 100.0)

    # Extra reduction from cutting N hours/day of average usage entirely
    avg_hourly = current_total / max(1, len(df))
    hours_saved_total = avg_hourly * hours_reduced_per_day * n_days

    total_saved = peak_saved + off_peak_saved + hours_saved_total
    total_saved = min(total_saved, current_total * 0.95)  # sanity cap

    scenario_total = current_total - total_saved
    scenario_daily = scenario_total / n_days

    pct_reduction = (total_saved / current_total * 100) if current_total else 0
    projected_monthly_saving = (total_saved / n_days) * 30

    current_monthly_cost = estimate_cost(current_daily * 30, state, custom_rate)
    scenario_monthly_cost = estimate_cost(scenario_daily * 30, state, custom_rate)

    return {
        "current_daily_kwh": round(current_daily, 3),
        "scenario_daily_kwh": round(scenario_daily, 3),
        "kwh_saved_per_day": round(total_saved / n_days, 3),
        "pct_reduction": round(pct_reduction, 1),
        "projected_monthly_saving_kwh": round(projected_monthly_saving, 2),
        "current_monthly_cost": current_monthly_cost["estimated_cost"],
        "scenario_monthly_cost": scenario_monthly_cost["estimated_cost"],
        "assumptions": [
            f"Peak hours assumed 18:00-22:59 ({peak_reduction_pct}% reduction applied there).",
            f"Off-peak reduction applied: {off_peak_reduction_pct}%.",
            f"Additional {hours_reduced_per_day} hrs/day of average-load usage assumed cut entirely.",
            current_monthly_cost["rate_basis"],
        ],
    }
