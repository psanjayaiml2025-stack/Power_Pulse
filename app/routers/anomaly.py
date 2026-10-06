from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models
from .data_access import get_hourly_df, get_daily_df
from ..ml.anomaly import detect_anomalies
import json

router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])


@router.get("")
def get_anomalies(dataset_id: int | None = None, db: Session = Depends(get_db)):
    hourly = get_hourly_df(db, dataset_id)
    daily = get_daily_df(hourly)
    try:
        result = detect_anomalies(daily)
    except ValueError as e:
        raise HTTPException(400, str(e))

    db.add(models.AnalysisHistory(
        dataset_id=dataset_id, analysis_type="anomaly",
        summary=f"{result['anomalies_found']} anomalies found",
        metrics_json=json.dumps({"count": result["anomalies_found"]}),
    ))
    db.commit()
    return result
