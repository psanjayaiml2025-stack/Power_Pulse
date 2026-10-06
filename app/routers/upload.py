"""Current data upload + validation + data-quality report - Section 5."""
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from .. import models
from ..ml.data_pipeline import validate_and_clean_upload

router = APIRouter(prefix="/api/upload", tags=["upload"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/current-data")
def upload_current_data(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Only CSV or Excel files are supported.")

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cleaned_df, report = validate_and_clean_upload(dest_path)

    cleaned_path = None
    if cleaned_df is not None:
        cleaned_path = os.path.join(UPLOAD_DIR, f"cleaned_{file.filename}.csv")
        cleaned_df.to_csv(cleaned_path, index=False)

    ds = models.Dataset(
        filename=file.filename,
        stored_path=cleaned_path or dest_path,
        kind="current",
        row_count=report["row_count"],
        start_date=report["start_date"],
        end_date=report["end_date"],
        missing_values=report["missing_values"],
        duplicate_rows=report["duplicate_rows"],
        quality_notes="; ".join(report["warnings"]),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    report["dataset_id"] = ds.id
    return report
