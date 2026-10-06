"""Main consumption graph data - Daily/Weekly/Monthly toggle, actual vs.
Indian-baseline overlay. Baseline series is aligned by calendar position
(not date, since it's a different set of households/years) so it's shown
as a comparable shape, clearly labeled. a non-Indian legacy dataset is never used here."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..ml.data_pipeline import resample_by_granularity
from .data_access import get_hourly_df, get_baseline_hourly

router = APIRouter(prefix="/api/trend", tags=["trend"])


@router.get("")
def get_trend(dataset_id: int | None = None, granularity: str = "daily", db: Session = Depends(get_db)):
    if granularity not in ("daily", "weekly", "monthly"):
        raise HTTPException(400, "granularity must be daily, weekly, or monthly.")

    current_hourly = get_hourly_df(db, dataset_id)
    current_series = resample_by_granularity(current_hourly, granularity)

    baseline_series = []
    baseline_hourly, baseline_label = get_baseline_hourly()
    if baseline_hourly is not None:
        baseline_df = resample_by_granularity(baseline_hourly, granularity)
        baseline_series = baseline_df.to_dict(orient="records")

    return {
        "granularity": granularity,
        "has_current_data": dataset_id is not None,
        "current": current_series.to_dict(orient="records"),
        "baseline": baseline_series,
        "baseline_label": baseline_label or "No baseline dataset available",
    }
