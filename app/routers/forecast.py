from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models
from .data_access import get_hourly_df, get_daily_df
from ..ml.forecasting import train_and_forecast
import json

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


@router.get("")
def get_forecast(dataset_id: int | None = None, horizon_days: int = 7, db: Session = Depends(get_db)):
    if horizon_days < 1 or horizon_days > 30:
        raise HTTPException(400, "horizon_days must be between 1 and 30.")
    hourly = get_hourly_df(db, dataset_id)
    daily = get_daily_df(hourly)
    try:
        result = train_and_forecast(daily, horizon_days)
    except ValueError as e:
        raise HTTPException(400, str(e))

    db.add(models.AnalysisHistory(
        dataset_id=dataset_id, analysis_type="forecast",
        summary=f"Forecast {horizon_days}d using {result['model_used']}",
        metrics_json=json.dumps(result["evaluation"]),
    ))
    db.commit()
    return result
