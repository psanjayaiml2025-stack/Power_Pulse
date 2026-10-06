from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..schemas import CalculatorRequest, ApplianceEstimateRequest
from .data_access import get_hourly_df, get_daily_df
from ..ml.tariffs import estimate_cost, estimate_appliance_usage, ILLUSTRATIVE_SLABS
import pandas as pd

router = APIRouter(prefix="/api/calculator", tags=["calculator"])


@router.get("/states")
def list_states():
    return list(ILLUSTRATIVE_SLABS.keys())


@router.post("")
def post_calculator(req: CalculatorRequest, db: Session = Depends(get_db)):
    hourly = get_hourly_df(db, req.dataset_id)
    daily = get_daily_df(hourly)

    n_days = max(1, len(daily))
    total_kwh = float(daily["consumption_kwh"].sum())
    avg_daily = total_kwh / n_days
    avg_weekly = avg_daily * 7
    avg_monthly = avg_daily * 30
    peak_day = daily.loc[daily["consumption_kwh"].idxmax()]

    hourly_by_hour = hourly.copy()
    hourly_by_hour["hour"] = pd.to_datetime(hourly_by_hour["timestamp"]).dt.hour
    peak_hour = hourly_by_hour.groupby("hour")["consumption_kwh"].mean().idxmax()

    first_half = daily.iloc[: n_days // 2]["consumption_kwh"].mean() if n_days >= 4 else None
    second_half = daily.iloc[n_days // 2 :]["consumption_kwh"].mean() if n_days >= 4 else None
    change_pct = None
    if first_half and second_half and first_half > 0:
        change_pct = round((second_half - first_half) / first_half * 100, 1)

    cost = estimate_cost(avg_monthly, req.state, req.custom_rate_per_kwh)

    return {
        "daily_kwh": round(avg_daily, 3),
        "weekly_kwh": round(avg_weekly, 3),
        "monthly_kwh": round(avg_monthly, 3),
        "peak_day": {"date": str(peak_day["date"]), "kwh": round(float(peak_day["consumption_kwh"]), 3)},
        "peak_hour_of_day": int(peak_hour),
        "consumption_change_pct_first_vs_second_half": change_pct,
        "estimated_monthly_cost": cost,
    }


@router.post("/appliance-estimate")
def post_appliance_estimate(req: ApplianceEstimateRequest):
    """Section 17 - independent from any meter/dataset. Pure watts x
    hours x days math, clearly labeled as an estimate."""
    return estimate_appliance_usage(
        req.watts, req.hours_per_day, req.days_per_month, req.state, req.custom_rate_per_kwh
    )
