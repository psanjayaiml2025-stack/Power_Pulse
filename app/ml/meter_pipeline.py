"""
EB Meter / Bill data pipeline.

Realistic household input: either
  (a) cumulative meter readings taken on different dates (like an odometer -
      consumption = difference between consecutive readings), or
  (b) units-consumed figures taken directly off bills (most Indian EB bills
      state "units consumed" for the billing period directly).

This module NEVER claims appliance-level or sub-metered detail - a
standard EB meter/bill only ever gives a single aggregate total. That
honesty rule is enforced here, not bolted on in the API layer.
"""
import pandas as pd
import numpy as np

_DATE_CANDIDATES = ["entry_date", "date", "reading_date", "bill_date", "billing_date"]
_UNITS_CANDIDATES = ["units", "units_consumed", "consumption", "consumption_kwh", "kwh", "energy"]
_CUMULATIVE_CANDIDATES = ["reading", "meter_reading", "cumulative", "cumulative_reading", "meter_value"]
_AMOUNT_CANDIDATES = ["amount", "bill_amount", "billed_amount", "cost", "bill"]


def _detect_column(columns, candidates):
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for cand in candidates:
        for lc, orig in lower_map.items():
            if cand in lc:
                return orig
    return None


def validate_and_clean_bill_upload(file_path: str):
    """Parses a bill/meter-reading CSV or Excel file. Auto-detects a date
    column plus EITHER a units-consumed column OR a cumulative-meter-reading
    column (never both silently - if both exist, units-consumed wins since
    it's what most bills actually print). Returns (rows, quality_report)."""
    warnings = []

    if file_path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file_path)
    else:
        raw = pd.read_csv(file_path)

    date_col = _detect_column(raw.columns, _DATE_CANDIDATES)
    units_col = _detect_column(raw.columns, _UNITS_CANDIDATES)
    cum_col = _detect_column(raw.columns, _CUMULATIVE_CANDIDATES)
    amount_col = _detect_column(raw.columns, _AMOUNT_CANDIDATES)

    detected = {
        "date_column": date_col,
        "units_column": units_col,
        "cumulative_reading_column": cum_col if not units_col else None,
        "amount_column": amount_col,
    }

    if date_col is None or (units_col is None and cum_col is None):
        return [], {
            "filename": file_path.split("/")[-1],
            "row_count": int(len(raw)),
            "rows_accepted": 0,
            "detected_columns": detected,
            "warnings": [
                "Could not detect a date column and a units/reading column. "
                "Expected columns like 'date' + 'units_consumed' (from your "
                "bill) or 'date' + 'meter_reading' (cumulative meter value)."
            ],
        }

    reading_type = "units" if units_col else "cumulative"
    value_col = units_col or cum_col

    df = raw[[date_col, value_col] + ([amount_col] if amount_col else [])].copy()
    cols = ["entry_date", "raw_value"] + (["billed_amount"] if amount_col else [])
    df.columns = cols

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    n_bad_dates = df["entry_date"].isna().sum()
    if n_bad_dates:
        warnings.append(f"{n_bad_dates} rows had unparseable dates and were dropped.")
    df = df.dropna(subset=["entry_date"])

    df["raw_value"] = pd.to_numeric(df["raw_value"], errors="coerce")
    n_bad_val = df["raw_value"].isna().sum()
    if n_bad_val:
        warnings.append(f"{n_bad_val} rows had non-numeric readings and were dropped.")
    df = df.dropna(subset=["raw_value"])

    n_negative = (df["raw_value"] < 0).sum()
    if n_negative:
        warnings.append(f"{n_negative} rows had negative values and were removed.")
        df = df[df["raw_value"] >= 0]

    df = df.sort_values("entry_date").drop_duplicates(subset=["entry_date"], keep="first")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "entry_date": r["entry_date"].strftime("%Y-%m-%d"),
            "reading_type": reading_type,
            "raw_value": float(r["raw_value"]),
            "billed_amount": float(r["billed_amount"]) if amount_col and not pd.isna(r["billed_amount"]) else None,
        })

    return rows, {
        "filename": file_path.split("/")[-1],
        "row_count": int(len(raw)),
        "rows_accepted": len(rows),
        "detected_columns": detected,
        "reading_type_used": reading_type,
        "warnings": warnings if warnings else ["No major data quality issues detected."],
    }


def compute_series(rows: list[dict]):
    """rows: list of {id, entry_date, reading_type, raw_value, billed_amount}
    sorted by nothing in particular - this function sorts them.
    Returns (series_df, warnings) where series_df has one row per PERIOD
    (i.e. one fewer than input rows for a pure-cumulative series, since the
    first cumulative reading has no prior point to diff against)."""
    warnings = []
    if not rows:
        return pd.DataFrame(columns=["id", "date", "units_consumed", "period_days", "daily_rate", "billed_amount"]), warnings

    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    records = []
    last_cumulative = None
    last_date = None
    for _, row in df.iterrows():
        consumed = None
        period_days = (row["entry_date"] - last_date).days if last_date is not None else None

        if row["reading_type"] == "units":
            consumed = row["raw_value"]
        else:  # cumulative
            if last_cumulative is not None:
                diff = row["raw_value"] - last_cumulative
                if diff < 0:
                    warnings.append(
                        f"Meter reading on {row['entry_date'].date()} is lower than the "
                        f"previous reading - possible meter reset/replacement. Skipped."
                    )
                else:
                    consumed = diff
            last_cumulative = row["raw_value"]

        if consumed is not None and period_days and period_days > 0:
            records.append({
                "id": int(row["id"]),
                "date": row["entry_date"].strftime("%Y-%m-%d"),
                "units_consumed": round(float(consumed), 2),
                "period_days": int(period_days),
                "daily_rate": round(float(consumed) / period_days, 3),
                "billed_amount": row.get("billed_amount") if pd.notna(row.get("billed_amount")) else None,
            })
        elif period_days is not None and period_days <= 0:
            warnings.append(f"Reading on {row['entry_date'].date()} is on/before the previous date - skipped.")

        last_date = row["entry_date"]

    return pd.DataFrame(records), warnings
