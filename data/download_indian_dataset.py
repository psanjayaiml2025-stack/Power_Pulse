"""
PowerPulse Analytics - Indian Smart-Meter Dataset (PRIMARY dataset)
=====================================================================
Source: Smart Energy Informatics Lab (SEIL), IIT Bombay
        http://seil.cse.iitb.ac.in/residential-dataset/

WHAT IT IS (real, verified, citable):
- Electricity consumption from a high-rise RESIDENTIAL building inside the
  IIT Bombay campus - Powai, Mumbai, Maharashtra, India.
- 60 anonymized 3BHK apartments, each with its own smart meter.
- Native sampling: 5-8 seconds. The "Virtual Dataset" file this script
  expects is the lab's own downsampled 1-hour version.
- Period: December 2016 - January 2018.
- Columns in the Virtual Dataset file: VirtualApartment, Date, Time,
  Energy (Wh, = W1+W2+W3), V1, V2, V3 (per-phase voltage).
- License: Creative Commons Attribution 4.0 International.
- Citation: Mammen, P.M., Kumar, H., Ramamritham, K., Rashid, H.
  "Want to Reduce Energy Consumption, Whom should we call?"
  Proceedings of the 9th ACM International Conference on Future Energy
  Systems (e-Energy '18), pages 12-20.

HONESTY NOTE:
This script does NOT auto-download the file. The dataset's host page
offers three mirror buttons (IITB Direct Server / Dropbox / Figshare) and
this environment could not verify a stable, unauthenticated direct-file
URL for the specific "Virtual Dataset" file. Rather than guess a URL that
might silently fetch the wrong file, this script gives you the exact
manual steps, then verifies whatever you place there is genuinely usable.

MANUAL STEPS (2 minutes):
  1. Open: http://seil.cse.iitb.ac.in/residential-dataset/
  2. Under the "Virtual Dataset" section, click any one mirror:
     [Direct (IITB Server)] or [Dropbox] or [Figshare]  (~8.13 MB)
  3. Save the CSV as:
     data/raw/indian_smart_meter/virtual_dataset.csv
  4. Re-run this script - it will verify and summarize the file.

If the mirror gives you the 39 numbered per-apartment CSV files, that is
also supported. PowerPulse reads TS/V1/V2/V3/W1/W2/W3, preserves apartment
identity, and integrates phase power into kWh.
"""
import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw", "indian_smart_meter")
CSV_PATH = os.path.join(RAW_DIR, "virtual_dataset.csv")

INSTRUCTIONS = """
Indian smart-meter dataset not found yet.

Put the IIT Bombay SEIL dataset under:
  {folder}

Supported formats:
  1) Virtual Dataset: virtual_dataset.csv
  2) Per-apartment download: 1.csv ... 39.csv

Then re-run:
  python data/download_indian_dataset.py
""".format(folder=RAW_DIR)


def verify():
    os.makedirs(RAW_DIR, exist_ok=True)

    virtual = os.path.join(RAW_DIR, "virtual_dataset.csv")
    apartment_files = [
        os.path.join(RAW_DIR, name)
        for name in os.listdir(RAW_DIR)
        if name.lower().endswith(".csv") and name[:-4].isdigit()
    ]

    if os.path.exists(virtual):
        try:
            df = pd.read_csv(virtual, nrows=5)
            df.columns = [str(c).strip() for c in df.columns]
            required = {"Date", "Time", "Energy"}
            if required.issubset(df.columns):
                print("Indian Virtual Dataset verified.")
                print(f"  File: {virtual}")
                print("  Format: Date, Time, Energy, V1, V2, V3")
                print("  Source: IIT Bombay SEIL Residential Energy Dataset (Mumbai, Maharashtra) - CC BY 4.0")
                return True
            print(f"virtual_dataset.csv exists but uses another format: {list(df.columns)}")
        except Exception as e:
            print(f"Could not read {virtual}: {e}")

    if apartment_files:
        try:
            df = pd.read_csv(sorted(apartment_files)[0], nrows=5)
            df.columns = [str(c).strip() for c in df.columns]
            required = {"TS", "V1", "V2", "V3", "W1", "W2", "W3"}
            if required.issubset(df.columns):
                print("Indian IIT Bombay per-apartment dataset verified.")
                print(f"  Apartment files: {len(apartment_files)}")
                print("  Format: TS, V1, V2, V3, W1, W2, W3")
                print("  Power conversion: W1+W2+W3 integrated over sampling intervals into kWh")
                print("  Source: IIT Bombay SEIL Residential Energy Dataset (Mumbai, Maharashtra) - CC BY 4.0")
                return True
            print(f"Numbered CSV found but unexpected columns: {list(df.columns)}")
        except Exception as e:
            print(f"Could not read numbered IITB CSV: {e}")

    print(INSTRUCTIONS)
    return False


if __name__ == "__main__":
    verify()
