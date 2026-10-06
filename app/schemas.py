"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class DataQualityReport(BaseModel):
    filename: str
    row_count: int
    missing_values: int
    duplicate_rows: int
    start_date: Optional[str]
    end_date: Optional[str]
    detected_columns: Dict[str, Optional[str]]
    unit_assumption: str
    warnings: List[str]
    dataset_id: int


class EDASummary(BaseModel):
    row_count: int
    feature_count: int
    missing_values: Dict[str, int]
    descriptive_stats: Dict[str, Dict[str, float]]
    hourly_pattern: List[Dict[str, Any]]
    daily_pattern: List[Dict[str, Any]]
    monthly_pattern: List[Dict[str, Any]]
    weekday_vs_weekend: Dict[str, float]
    top_correlations: List[Dict[str, Any]]


class ForecastRequest(BaseModel):
    dataset_id: Optional[int] = None
    horizon_days: int = 7


class ForecastResponse(BaseModel):
    model_used: str
    horizon_days: int
    evaluation: Dict[str, float]
    baseline_evaluation: Dict[str, float]
    history: List[Dict[str, Any]]
    forecast: List[Dict[str, Any]]
    feature_importance: List[Dict[str, Any]]


class AnomalyResponse(BaseModel):
    method: str
    total_points: int
    anomalies_found: int
    anomalies: List[Dict[str, Any]]


class WhatIfRequest(BaseModel):
    dataset_id: Optional[int] = None
    peak_reduction_pct: float = 0.0
    off_peak_reduction_pct: float = 0.0
    hours_reduced_per_day: float = 0.0
    state: str = "Tamil Nadu"
    custom_rate_per_kwh: Optional[float] = None


class WhatIfResponse(BaseModel):
    current_daily_kwh: float
    scenario_daily_kwh: float
    kwh_saved_per_day: float
    pct_reduction: float
    projected_monthly_saving_kwh: float
    current_monthly_cost: Optional[float]
    scenario_monthly_cost: Optional[float]
    assumptions: List[str]


class CalculatorRequest(BaseModel):
    dataset_id: Optional[int] = None
    state: str = "Tamil Nadu"
    custom_rate_per_kwh: Optional[float] = None


class ApplianceEstimateRequest(BaseModel):
    watts: float
    hours_per_day: float
    days_per_month: float = 30
    state: str = "Tamil Nadu"
    custom_rate_per_kwh: Optional[float] = None


class MeterReadingCreate(BaseModel):
    entry_date: str  # "YYYY-MM-DD"
    reading_type: str  # "cumulative" | "units"
    value: float
    billed_amount: Optional[float] = None


class MeterWhatIfRequest(BaseModel):
    reduction_pct: float = 10.0
    state: str = "Tamil Nadu"
    custom_rate_per_kwh: Optional[float] = None
    next_period_days: int = 30


class MeterQuickCalcRequest(BaseModel):
    previous_reading: float
    previous_date: str
    current_reading: float
    current_date: str
    state: str = "Tamil Nadu"
    custom_rate_per_kwh: Optional[float] = None


class Recommendation(BaseModel):
    observation: str
    evidence: str
    suggested_action: str
    expected_impact: str
    severity: str
