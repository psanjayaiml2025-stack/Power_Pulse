from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..schemas import WhatIfRequest
from .data_access import get_hourly_df
from ..ml.whatif import run_whatif

router = APIRouter(prefix="/api/whatif", tags=["whatif"])


@router.post("")
def post_whatif(req: WhatIfRequest, db: Session = Depends(get_db)):
    hourly = get_hourly_df(db, req.dataset_id)
    return run_whatif(
        hourly, req.peak_reduction_pct, req.off_peak_reduction_pct,
        req.hours_reduced_per_day, req.state, req.custom_rate_per_kwh,
    )
