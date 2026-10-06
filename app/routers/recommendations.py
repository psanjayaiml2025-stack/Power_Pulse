from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models
from .data_access import get_hourly_df, get_daily_df
from ..ml.anomaly import detect_anomalies
from ..ml.recommendations import generate_recommendations
import json

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.get("")
def get_recommendations(dataset_id: int | None = None, db: Session = Depends(get_db)):
    hourly = get_hourly_df(db, dataset_id)
    daily = get_daily_df(hourly)
    try:
        anomaly_result = detect_anomalies(daily)
        anomalies = anomaly_result["anomalies"]
    except ValueError:
        anomalies = []

    recs = generate_recommendations(hourly, daily, anomalies)

    db.add(models.AnalysisHistory(
        dataset_id=dataset_id, analysis_type="recommendation",
        summary=f"{len(recs)} recommendations generated",
        metrics_json=json.dumps({"count": len(recs)}),
    ))
    db.commit()
    return recs
