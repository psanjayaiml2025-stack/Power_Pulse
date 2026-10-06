"""
EB Meter / Bill readings - the realistic primary input for a household
without a smart meter: periodic bill units or manual meter readings.
"""
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from .. import models
from ..schemas import MeterReadingCreate, MeterWhatIfRequest, MeterQuickCalcRequest
from ..ml.meter_pipeline import validate_and_clean_bill_upload, compute_series
from ..ml.meter_analytics import (
    eda_summary, detect_meter_anomalies, forecast_next_period,
    whatif_meter, generate_meter_recommendations, build_meter_dashboard,
)
from ..ml.data_pipeline import indian_dataset_available, load_indian_hourly, resample_daily

router = APIRouter(prefix="/api/meter", tags=["meter"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _get_series(db: Session):
    rows = [
        {"id": r.id, "entry_date": r.entry_date, "reading_type": r.reading_type,
         "raw_value": r.raw_value, "billed_amount": r.billed_amount}
        for r in db.query(models.MeterReading).all()
    ]
    return compute_series(rows)


def _benchmark_daily_rate():
    """Reference-only average daily kWh from the Indian smart-meter
    dataset, for context - clearly not the user's own household.
    Never uses a non-Indian legacy dataset."""
    if not indian_dataset_available():
        return None
    try:
        daily = resample_daily(load_indian_hourly())
        return float(daily["consumption_kwh"].mean())
    except Exception:
        return None


@router.post("/quick-calculate")
def quick_calculate(payload: MeterQuickCalcRequest):
    """One-off previous-vs-current reading calculation - matches the
    exact 'Previous Reading / Current Reading' workflow from the spec.
    Does not touch the database; pure calculation for a quick estimate."""
    from datetime import datetime as dt
    from ..ml.tariffs import estimate_cost

    try:
        prev_date = dt.strptime(payload.previous_date, "%Y-%m-%d")
        curr_date = dt.strptime(payload.current_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Dates must be in YYYY-MM-DD format.")

    days = (curr_date - prev_date).days
    if days <= 0:
        raise HTTPException(400, "Current reading date must be after the previous reading date.")

    units_consumed = payload.current_reading - payload.previous_reading
    if units_consumed < 0:
        raise HTTPException(
            400,
            "Current reading is lower than the previous reading - check for a "
            "meter reset/replacement or a data entry error."
        )

    avg_per_day = units_consumed / days
    estimated_monthly = avg_per_day * 30
    cost = estimate_cost(estimated_monthly, payload.state, payload.custom_rate_per_kwh)

    return {
        "units_consumed": round(units_consumed, 2),
        "days": days,
        "avg_units_per_day": round(avg_per_day, 3),
        "estimated_monthly_consumption": round(estimated_monthly, 2),
        "estimated_monthly_cost": cost,
    }


@router.post("/reading")
def add_reading(payload: MeterReadingCreate, db: Session = Depends(get_db)):
    if payload.reading_type not in ("cumulative", "units"):
        raise HTTPException(400, "reading_type must be 'cumulative' or 'units'.")
    row = models.MeterReading(
        entry_date=payload.entry_date, reading_type=payload.reading_type,
        raw_value=payload.value, billed_amount=payload.billed_amount, source="manual",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    series, warnings = _get_series(db)
    return {
        "reading_id": row.id,
        "total_readings": db.query(models.MeterReading).count(),
        "computed_periods": len(series),
        "warnings": warnings,
        "latest_periods": series.tail(5).to_dict(orient="records"),
    }


@router.post("/upload")
def upload_bill_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Only CSV or Excel files are supported.")

    dest_path = os.path.join(UPLOAD_DIR, f"bill_{file.filename}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    rows, report = validate_and_clean_bill_upload(dest_path)
    for r in rows:
        db.add(models.MeterReading(
            entry_date=r["entry_date"], reading_type=r["reading_type"],
            raw_value=r["raw_value"], billed_amount=r["billed_amount"], source="upload",
        ))
    db.commit()
    report["inserted"] = len(rows)
    return report


@router.get("/readings")
def list_readings(db: Session = Depends(get_db)):
    rows = db.query(models.MeterReading).order_by(models.MeterReading.entry_date).all()
    raw = [
        {"id": r.id, "entry_date": r.entry_date, "reading_type": r.reading_type,
         "raw_value": r.raw_value, "billed_amount": r.billed_amount, "source": r.source}
        for r in rows
    ]
    series, warnings = compute_series(raw)
    return {"raw_readings": raw, "computed_periods": series.to_dict(orient="records"), "warnings": warnings}


@router.delete("/reading/{reading_id}")
def delete_reading(reading_id: int, db: Session = Depends(get_db)):
    row = db.query(models.MeterReading).filter(models.MeterReading.id == reading_id).first()
    if not row:
        raise HTTPException(404, "Reading not found.")
    db.delete(row)
    db.commit()
    return {"deleted": reading_id}


@router.get("/eda")
def get_meter_eda(db: Session = Depends(get_db)):
    series, _ = _get_series(db)
    return eda_summary(series)


@router.get("/anomaly")
def get_meter_anomaly(db: Session = Depends(get_db)):
    series, _ = _get_series(db)
    return detect_meter_anomalies(series)


@router.get("/forecast")
def get_meter_forecast(next_period_days: int = 30, db: Session = Depends(get_db)):
    series, _ = _get_series(db)
    return forecast_next_period(series, next_period_days)


@router.post("/whatif")
def post_meter_whatif(payload: MeterWhatIfRequest, db: Session = Depends(get_db)):
    series, _ = _get_series(db)
    return whatif_meter(series, payload.reduction_pct, payload.state, payload.custom_rate_per_kwh, payload.next_period_days)


@router.get("/recommendations")
def get_meter_recommendations(db: Session = Depends(get_db)):
    series, _ = _get_series(db)
    anomaly_result = detect_meter_anomalies(series)
    return generate_meter_recommendations(series, anomaly_result)


@router.get("/dashboard")
def get_meter_dashboard(state: str = "Tamil Nadu", custom_rate_per_kwh: float | None = None, db: Session = Depends(get_db)):
    series, _ = _get_series(db)
    return build_meter_dashboard(series, state, custom_rate_per_kwh, _benchmark_daily_rate())
