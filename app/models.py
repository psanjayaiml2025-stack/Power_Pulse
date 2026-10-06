"""SQLAlchemy ORM models. Raw datasets stay on disk (data/); the DB only
stores metadata, analysis history, and reports - per spec Section 23."""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime, timezone
from .database import Base


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)
    kind = Column(String, default="current")  # "current" (user upload) | "historical"
    row_count = Column(Integer, default=0)
    start_date = Column(String, nullable=True)
    end_date = Column(String, nullable=True)
    missing_values = Column(Integer, default=0)
    duplicate_rows = Column(Integer, default=0)
    quality_notes = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, nullable=True)
    analysis_type = Column(String)  # eda | forecast | anomaly | whatif | recommendation
    summary = Column(Text)
    metrics_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    content_json = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Setting(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String)


class MeterReading(Base):
    """A single EB meter reading or bill entry - the realistic primary
    input for a household with no smart meter. Only raw facts are stored;
    per-period consumption (units_consumed) is derived on read, not
    stored, so it always reflects the current full reading history."""
    __tablename__ = "meter_readings"
    id = Column(Integer, primary_key=True, index=True)
    entry_date = Column(String, nullable=False)  # ISO date "YYYY-MM-DD"
    reading_type = Column(String, nullable=False)  # "cumulative" | "units"
    raw_value = Column(Float, nullable=False)
    billed_amount = Column(Float, nullable=True)
    source = Column(String, default="manual")  # "manual" | "upload"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
