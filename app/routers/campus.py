"""Building/Meter hierarchy endpoint - College/Campus and Institution mode."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from .data_access import get_hourly_df
from ..ml.campus import building_breakdown

router = APIRouter(prefix="/api/campus", tags=["campus"])


@router.get("/buildings")
def get_buildings(dataset_id: int | None = None, db: Session = Depends(get_db)):
    hourly = get_hourly_df(db, dataset_id)
    return building_breakdown(hourly)
