from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/analyses")
def list_analyses(db: Session = Depends(get_db)):
    rows = db.query(models.AnalysisHistory).order_by(models.AnalysisHistory.created_at.desc()).limit(50).all()
    return [
        {
            "id": r.id, "dataset_id": r.dataset_id, "analysis_type": r.analysis_type,
            "summary": r.summary, "created_at": r.created_at.isoformat(),
        } for r in rows
    ]


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    rows = db.query(models.Dataset).order_by(models.Dataset.uploaded_at.desc()).all()
    return [
        {
            "id": d.id, "filename": d.filename, "kind": d.kind, "row_count": d.row_count,
            "start_date": d.start_date, "end_date": d.end_date,
            "missing_values": d.missing_values, "duplicate_rows": d.duplicate_rows,
            "quality_notes": d.quality_notes, "uploaded_at": d.uploaded_at.isoformat(),
        } for d in rows
    ]
