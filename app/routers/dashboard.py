"""Consolidated Energy Intelligence Dashboard endpoint - one call powers
the whole Overview page (current vs baseline, where-is-my-energy-going,
why-did-it-increase, if-I-continue-like-this).

Baseline is always the Indian smart-meter dataset (or the user's own data
self-compared if that dataset isn't downloaded yet) - never a non-Indian legacy dataset."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .data_access import get_hourly_df, get_daily_df, get_baseline_hourly
from ..ml.insights import build_dashboard

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("")
def get_dashboard(dataset_id: int | None = None, state: str = "Tamil Nadu",
                   custom_rate_per_kwh: float | None = None, db: Session = Depends(get_db)):
    baseline_hourly, baseline_label = get_baseline_hourly()
    baseline_daily = get_daily_df(baseline_hourly) if baseline_hourly is not None else None

    has_current = dataset_id is not None
    if has_current:
        current_hourly = get_hourly_df(db, dataset_id)
    elif baseline_hourly is not None:
        current_hourly = baseline_hourly
    else:
        raise HTTPException(
            400,
            "No Indian smart-meter dataset found and no dataset uploaded. "
            "Run `python data/download_indian_dataset.py` or upload/enter your own data."
        )
    current_daily = get_daily_df(current_hourly)

    if baseline_hourly is None:
        baseline_hourly, baseline_daily = current_hourly, current_daily

    result = build_dashboard(
        current_hourly, current_daily, baseline_hourly, baseline_daily,
        has_current, state, custom_rate_per_kwh,
    )
    result["baseline_label"] = baseline_label
    return result
