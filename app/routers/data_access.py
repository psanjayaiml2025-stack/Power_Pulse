"""Shared helper to load a dataset's hourly dataframe.

PRIMARY default (no dataset_id given): the Indian smart-meter dataset.
No non-Indian legacy dataset is used as a fallback."""
import os
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session
from .. import models
from ..ml.data_pipeline import (
    load_indian_hourly, indian_dataset_available, INDIAN_LABEL,
    add_time_features, resample_daily,
)


def get_hourly_df(db: Session, dataset_id: int | None):
    if dataset_id is None:
        if not indian_dataset_available():
            raise HTTPException(
                400,
                "No Indian smart-meter dataset found and no dataset_id given. "
                "Run `python data/download_indian_dataset.py` (see its printed "
                "instructions), or upload/enter your own current data first."
            )
        df = load_indian_hourly()
        return add_time_features(df)

    ds = db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} not found.")
    if not os.path.exists(ds.stored_path):
        raise HTTPException(404, "Stored dataset file is missing on disk.")

    df = pd.read_csv(ds.stored_path, parse_dates=["timestamp"])
    return add_time_features(df)


def get_daily_df(hourly_df: pd.DataFrame):
    return resample_daily(hourly_df)


def get_baseline_hourly():
    """The comparison reference for insights uses the Indian dataset only."""
    if not indian_dataset_available():
        return None, None
    return add_time_features(load_indian_hourly()), INDIAN_LABEL
