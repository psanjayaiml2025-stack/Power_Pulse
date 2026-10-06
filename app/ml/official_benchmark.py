"""Official CEA benchmark data packaged with PowerPulse.

This layer is contextual/benchmark data. It is intentionally separate from
the IIT Bombay interval smart-meter training data.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "data" / "official" / "cea_2025" / "national_consumption_2023_24_2024_25.csv"

def get_official_benchmark():
    df = pd.read_csv(CSV_PATH)
    rows = df.to_dict(orient="records")
    total_23 = float(df.loc[df.category=="Total","fy_2023_24_gwh"].iloc[0])
    total_24 = float(df.loc[df.category=="Total","fy_2024_25_gwh"].iloc[0])
    growth = round((total_24-total_23)/total_23*100,2)
    return {
        "available": True,
        "source": "Central Electricity Authority (CEA), All India Electricity Statistics / Growth Book 2025",
        "coverage": "India, FY 2023-24 and provisional FY 2024-25",
        "official_url": "https://cea.nic.in/general-review-report/?lang=en",
        "report_url": "https://cea.nic.in/wp-content/uploads/pdm/2025/11/Growth_Book_2025.pdf",
        "total_2023_24_gwh": int(total_23),
        "total_2024_25_gwh": int(total_24),
        "total_growth_pct": growth,
        "categories": rows,
        "note": "Official 2024-25 values are provisional. This benchmark is not used as household smart-meter training data."
    }
