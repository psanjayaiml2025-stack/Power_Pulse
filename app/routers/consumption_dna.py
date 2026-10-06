from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from .data_access import get_hourly_df, get_daily_df
from ..ml.consumption_dna import compute_consumption_dna

router = APIRouter(prefix="/api/dna", tags=["consumption-dna"])


@router.get("")
def get_consumption_dna(dataset_id: int | None = None, db: Session = Depends(get_db)):
    hourly = get_hourly_df(db, dataset_id)
    daily = get_daily_df(hourly)
    return compute_consumption_dna(hourly, daily)
