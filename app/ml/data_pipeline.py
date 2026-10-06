"""
Data pipeline: loading, validation, cleaning, standardization.

Primary dataset: IIT Bombay SEIL Indian residential smart-meter data.
User uploads and bill/meter readings use the same internal consumption schema.
"""
import os
import pandas as pd
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- PRIMARY: Indian smart-meter dataset ----------------------------------
INDIAN_CSV = os.path.join(BASE_DIR, "data", "raw", "indian_smart_meter", "virtual_dataset.csv")
INDIAN_LABEL = (
    "Indian Smart-Meter Dataset — IIT Bombay Residential Campus "
    "(Powai, Mumbai, Maharashtra, India), Dec 2016–Jan 2018, "
    "60 anonymized apartments, 1-hour readings "
    "(Smart Energy Informatics Lab, IIT Bombay - CC BY 4.0)"
)

_TIMESTAMP_CANDIDATES = ["timestamp", "x_timestamp", "datetime", "date_time", "date", "time", "ts"]
_CONSUMPTION_CANDIDATES = [
    "consumption_kwh", "t_kwh", "kwh", "energy_kwh", "global_active_power",
    "power_kw", "consumption", "energy", "power", "units", "units_consumed",
]
_VOLTAGE_CANDIDATES = ["voltage", "z_avg voltage (volt)", "avg voltage (volt)", "voltage (v)", "volt"]
_CURRENT_CANDIDATES = ["current", "z_avg current (amp)", "avg current (amp)", "current (a)", "amp"]
_FREQUENCY_CANDIDATES = ["frequency", "y_freq (hz)", "freq (hz)", "freq", "hz"]
_METER_ID_CANDIDATES = ["meter_id", "meter", "meter id", "device_id", "meter identifier"]
_BUILDING_CANDIDATES = ["building", "block", "zone", "campus_building", "building_name", "location"]

# Documents the exact mapping described in the upgrade spec, so the report
# can show the user precisely what was matched to what.
COLUMN_MAP_EXAMPLES = {
    "x_Timestamp": "timestamp", "t_kWh": "consumption_kwh",
    "z_Avg Voltage (Volt)": "voltage", "z_Avg Current (Amp)": "current",
    "y_Freq (Hz)": "frequency", "meter": "meter_id",
}


def _indian_source_files():
    """Return the supported IIT Bombay Indian dataset source files.

    Supports both the documented Virtual Dataset CSV and the 39-file
    per-apartment download (1.csv ... 39.csv) that the IITB mirror can return.
    """
    folder = os.path.join(BASE_DIR, "data", "raw", "indian_smart_meter")
    virtual = os.path.join(folder, "virtual_dataset.csv")
    if os.path.isfile(virtual):
        return [virtual]
    files = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if re.fullmatch(r"\d+\.csv", name, flags=re.IGNORECASE):
            files.append(os.path.join(folder, name))
    return files


def indian_dataset_available() -> bool:
    return bool(_indian_source_files())


def _read_indian_virtual(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"Date", "Time", "Energy"}
    if not required.issubset(df.columns):
        raise ValueError(f"Virtual Dataset missing columns: {sorted(required - set(df.columns))}")
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
    )
    df["energy_wh"] = pd.to_numeric(df["Energy"], errors="coerce")
    if all(c in df.columns for c in ["V1", "V2", "V3"]):
        for c in ["V1", "V2", "V3"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["voltage"] = df[["V1", "V2", "V3"]].mean(axis=1)
    else:
        df["voltage"] = np.nan
    df["meter_id"] = "virtual_dataset"
    return df[["timestamp", "energy_wh", "voltage", "meter_id"]].dropna(subset=["timestamp", "energy_wh"])


def _read_indian_apartment(path: str) -> pd.DataFrame:
    """Read an IITB per-apartment file: TS,V1,V2,V3,W1,W2,W3.

    TS is Unix epoch seconds and W1/W2/W3 are phase power readings in watts.
    Energy is calculated by integrating power over each observed sampling
    interval, rather than treating watts as kWh.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"TS", "W1", "W2", "W3"}
    if not required.issubset(df.columns):
        raise ValueError(f"IITB apartment file missing columns: {sorted(required - set(df.columns))}")
    ts_num = pd.to_numeric(df["TS"], errors="coerce")
    df["timestamp"] = pd.to_datetime(ts_num, unit="s", errors="coerce", utc=True).dt.tz_convert(None)
    for c in ["W1", "W2", "W3", "V1", "V2", "V3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["power_w"] = df[["W1", "W2", "W3"]].fillna(0).sum(axis=1).clip(lower=0)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    dt_h = df["timestamp"].diff().dt.total_seconds().div(3600.0)
    positive = dt_h[(dt_h > 0) & (dt_h <= 1)]
    fallback_h = float(positive.median()) if len(positive) else (5.0 / 60.0)
    df["interval_h"] = dt_h.where((dt_h > 0) & (dt_h <= 1), fallback_h)
    df["energy_wh"] = df["power_w"] * df["interval_h"]
    if all(c in df.columns for c in ["V1", "V2", "V3"]):
        df["voltage"] = df[["V1", "V2", "V3"]].mean(axis=1)
    else:
        df["voltage"] = np.nan
    df["meter_id"] = os.path.splitext(os.path.basename(path))[0]
    return df[["timestamp", "energy_wh", "voltage", "meter_id"]]


def load_indian_hourly() -> pd.DataFrame:
    """Load the IIT Bombay Indian residential dataset.

    Accepts either the official Virtual Dataset (Date/Time/Energy) or the
    39 per-apartment CSV files (TS,V1,V2,V3,W1,W2,W3). For the latter, power
    is converted to kWh using the observed sampling interval, then the
    apartment readings are averaged at each timestamp to produce the
    representative household series used by the existing analytics pipeline.
    """
    files = _indian_source_files()
    if not files:
        raise FileNotFoundError(
            "Indian smart-meter dataset not found. Put the IITB Virtual Dataset "
            "or the numbered apartment CSVs (1.csv ... 39.csv) in "
            "data/raw/indian_smart_meter/."
        )

    frames = []
    virtual = len(files) == 1 and os.path.basename(files[0]).lower() == "virtual_dataset.csv"
    for path in files:
        frames.append(_read_indian_virtual(path) if virtual else _read_indian_apartment(path))

    df = pd.concat(frames, ignore_index=True)
    df["energy_wh"] = pd.to_numeric(df["energy_wh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "energy_wh"])
    df = df[df["energy_wh"] >= 0]

    # Preserve apartment identity so the Indian dataset can support
    # campus benchmarking and unsupervised behaviour clustering.
    df["consumption_kwh"] = df["energy_wh"] / 1000.0
    cols = ["timestamp", "consumption_kwh", "meter_id"]
    if df["voltage"].notna().any():
        cols.append("voltage")
    return df[cols].sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

def validate_and_clean_upload(file_path: str):
    """Loads a user-uploaded CSV/XLSX, auto-detects timestamp + consumption
    columns AND optional electrical fields (voltage/current/frequency/meter
    id) - e.g. an Indian smart-meter export with columns like
    'x_Timestamp', 't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)',
    'y_Freq (Hz)', 'meter'. Detected extra fields are mapped and KEPT
    (never dropped), never fabricated if absent. Returns
    (cleaned_df, quality_report_dict)."""
    warnings = []
    original_row_count = None

    if file_path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file_path)
    else:
        raw = pd.read_csv(file_path)
    original_row_count = len(raw)
    original_col_count = len(raw.columns)

    ts_col = _detect_column(raw.columns, _TIMESTAMP_CANDIDATES)
    val_col = _detect_column(raw.columns, _CONSUMPTION_CANDIDATES)
    # IIT Bombay per-apartment exports expose phase power as W1/W2/W3.
    # Convert their combined power to kWh using the observed sampling interval.
    if val_col is None and all(c in raw.columns for c in ["W1", "W2", "W3"]):
        ts_candidate = _detect_column(raw.columns, ["ts", "timestamp"])
        if ts_candidate is not None:
            raw["__iib_power_w"] = (
                pd.to_numeric(raw["W1"], errors="coerce").fillna(0)
                + pd.to_numeric(raw["W2"], errors="coerce").fillna(0)
                + pd.to_numeric(raw["W3"], errors="coerce").fillna(0)
            ).clip(lower=0)
            ts_num = pd.to_numeric(raw[ts_candidate], errors="coerce")
            raw["__iib_timestamp"] = pd.to_datetime(ts_num, unit="s", errors="coerce", utc=True).dt.tz_convert(None)
            raw = raw.sort_values("__iib_timestamp")
            dt_h = raw["__iib_timestamp"].diff().dt.total_seconds().div(3600.0)
            valid = dt_h[(dt_h > 0) & (dt_h <= 1)]
            fallback_h = float(valid.median()) if len(valid) else 5.0 / 60.0
            raw["__iib_energy_kwh"] = raw["__iib_power_w"] * dt_h.where((dt_h > 0) & (dt_h <= 1), fallback_h) / 1000.0
            raw["__iib_energy_kwh"] = raw["__iib_energy_kwh"].fillna(0)
            raw["__iib_ts_text"] = raw["__iib_timestamp"]
            ts_col = "__iib_ts_text"
            val_col = "__iib_energy_kwh"
        else:
            ts_col = None
    volt_col = _detect_column(raw.columns, _VOLTAGE_CANDIDATES)
    curr_col = _detect_column(raw.columns, _CURRENT_CANDIDATES)
    freq_col = _detect_column(raw.columns, _FREQUENCY_CANDIDATES)
    meter_col = _detect_column(raw.columns, _METER_ID_CANDIDATES)
    building_col = _detect_column(raw.columns, _BUILDING_CANDIDATES)

    detected_columns = {
        "timestamp_column": ts_col, "consumption_column": val_col,
        "voltage_column": volt_col, "current_column": curr_col,
        "frequency_column": freq_col, "meter_id_column": meter_col,
        "building_column": building_col,
    }

    if ts_col is None or val_col is None:
        return None, {
            "filename": os.path.basename(file_path),
            "row_count": original_row_count,
            "column_count": original_col_count,
            "missing_values": int(raw.isna().sum().sum()),
            "duplicate_rows": int(raw.duplicated().sum()),
            "start_date": None,
            "end_date": None,
            "detected_columns": detected_columns,
            "unit_assumption": "unknown",
            "sampling_frequency": None,
            "usable_pct": 0.0,
            "warnings": [
                "Could not confidently detect a timestamp and/or consumption "
                "column. Supported names include 'timestamp'/'x_Timestamp' "
                "and 'consumption_kwh'/'kwh'/'t_kWh'. Rename your columns "
                "and re-upload, or use Meter Readings & Bills instead."
            ],
        }

    keep_cols = {ts_col: "timestamp", val_col: "consumption_kwh"}
    if volt_col: keep_cols[volt_col] = "voltage"
    if curr_col: keep_cols[curr_col] = "current"
    if freq_col: keep_cols[freq_col] = "frequency"
    if meter_col: keep_cols[meter_col] = "meter_id"
    if building_col: keep_cols[building_col] = "building"

    df = raw[list(keep_cols.keys())].copy()
    df.columns = list(keep_cols.values())
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    n_bad_ts = int(df["timestamp"].isna().sum())
    if n_bad_ts:
        warnings.append(f"{n_bad_ts} rows had unparseable/invalid timestamps and were dropped.")
    df = df.dropna(subset=["timestamp"])

    df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
    n_bad_val = int(df["consumption_kwh"].isna().sum())
    if n_bad_val:
        warnings.append(f"{n_bad_val} rows had non-numeric/invalid consumption values and were dropped.")

    n_negative = int((df["consumption_kwh"] < 0).sum())
    if n_negative:
        warnings.append(f"{n_negative} rows had negative consumption and were removed.")
        df = df[df["consumption_kwh"] >= 0]

    # Coerce optional electrical fields to numeric without dropping rows over them
    for col in ["voltage", "current", "frequency"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_rows = int(df.duplicated(subset=["timestamp"]).sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate timestamp rows were found; kept the first.")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    missing_values = int(df["consumption_kwh"].isna().sum())

    # Heuristic unit check: if median value is enormous, likely already in
    # Wh rather than kWh - flag rather than silently rescale.
    unit_assumption = "kWh (as provided)"
    if len(df) and df["consumption_kwh"].median() > 500:
        unit_assumption = "possibly Wh, not kWh - values look unusually large"
        warnings.append(
            "Consumption values look unusually large for kWh - please verify "
            "your units before trusting cost estimates."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Real sampling frequency (median gap between consecutive readings)
    sampling_frequency = None
    if len(df) > 1:
        median_gap = df["timestamp"].diff().median()
        total_seconds = median_gap.total_seconds()
        if total_seconds < 90:
            sampling_frequency = f"~{int(total_seconds)}s (near real-time smart-meter)"
        elif total_seconds < 3600 * 1.5:
            sampling_frequency = f"~{round(total_seconds/60)} min"
        elif total_seconds < 3600 * 25:
            sampling_frequency = f"~{round(total_seconds/3600, 1)} hr"
        else:
            sampling_frequency = f"~{round(total_seconds/86400, 1)} days (bill/periodic readings)"

    usable_pct = round(len(df) / original_row_count * 100, 1) if original_row_count else 0.0

    extra_fields_detected = [v for k, v in keep_cols.items() if v not in ("timestamp", "consumption_kwh")]

    quality_report = {
        "filename": os.path.basename(file_path),
        "row_count": original_row_count,
        "column_count": original_col_count,
        "rows_usable": int(len(df)),
        "usable_pct": usable_pct,
        "missing_values": missing_values,
        "invalid_timestamps": n_bad_ts,
        "invalid_consumption_values": n_bad_val,
        "duplicate_rows": duplicate_rows,
        "start_date": str(df["timestamp"].min()) if len(df) else None,
        "end_date": str(df["timestamp"].max()) if len(df) else None,
        "sampling_frequency": sampling_frequency,
        "detected_columns": detected_columns,
        "extra_electrical_fields_detected": extra_fields_detected,
        "unit_assumption": unit_assumption,
        "warnings": warnings if warnings else ["No major data quality issues detected."],
    }
    return df, quality_report


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date
    return df


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("timestamp")[["consumption_kwh"]].resample("D").sum().reset_index()
    d = d.rename(columns={"timestamp": "date"})
    return d


def resample_by_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """granularity: 'daily' | 'weekly' | 'monthly' - for the main trend chart."""
    rule = {"daily": "D", "weekly": "W", "monthly": "MS"}.get(granularity, "D")
    d = df.set_index("timestamp")[["consumption_kwh"]].resample(rule).sum().reset_index()
    d = d.rename(columns={"timestamp": "period"})
    d["period"] = d["period"].dt.strftime("%Y-%m-%d")
    return d
