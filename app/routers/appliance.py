from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from .data_access import get_hourly_df
from ..ml.appliance import appliance_breakdown

router = APIRouter(prefix="/api/appliance", tags=["appliance"])


@router.get("")
def get_appliance(dataset_id: int | None = None, db: Session = Depends(get_db)):
    hourly = get_hourly_df(db, dataset_id)
    return appliance_breakdown(hourly)
