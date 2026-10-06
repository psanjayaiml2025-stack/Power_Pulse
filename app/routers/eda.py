from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from .data_access import get_hourly_df, get_daily_df
from ..ml.eda import build_eda

router = APIRouter(prefix="/api/eda", tags=["eda"])


@router.get("")
def get_eda(dataset_id: int | None = None, db: Session = Depends(get_db)):
    hourly = get_hourly_df(db, dataset_id)
    daily = get_daily_df(hourly)
    return build_eda(hourly, daily)
