from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .data_access import get_hourly_df
from ..database import get_db
from ..ml.clustering import cluster_energy_behaviour

router=APIRouter(prefix="/api/clustering",tags=["clustering"])

@router.get("")
def get_clustering(dataset_id:int|None=None,db:Session=Depends(get_db)):
    return cluster_energy_behaviour(get_hourly_df(db,dataset_id))
