"""
Appliance-level intelligence - Section 8 honesty rule.

Appliance-level intelligence is intentionally conservative. Standard
Indian EB bills and aggregate smart meters do not identify individual
appliances, so PowerPulse never invents appliance attribution.
"""
import pandas as pd


def appliance_breakdown(hourly: pd.DataFrame):
    has_submetering = all(
        c in hourly.columns for c in ["sub_kitchen_wh", "sub_laundry_wh", "sub_waterheater_ac_wh"]
    )
    if not has_submetering:
        return {
            "available": False,
            "message": (
                "A standard EB meter or electricity bill only ever reports a single "
                "aggregate total - it cannot distinguish individual appliances. "
                "PowerPulse will not fabricate an appliance breakdown from bill or "
                "single-column consumption data. This module only activates for "
                "datasets with real sub-metered channels (a source dataset with genuine sub-metered channels)."
            ),
            "categories": [],
        }

    kitchen_wh = hourly["sub_kitchen_wh"].sum()
    laundry_wh = hourly["sub_laundry_wh"].sum()
    wh_ac_wh = hourly["sub_waterheater_ac_wh"].sum()
    total_submetered_wh = kitchen_wh + laundry_wh + wh_ac_wh
    total_consumption_wh = hourly["consumption_kwh"].sum() * 1000

    other_wh = max(0, total_consumption_wh - total_submetered_wh)
    total = total_consumption_wh if total_consumption_wh else 1

    categories = [
        {"category": "Kitchen (dishwasher/oven/microwave)", "measured": True, "share_pct": round(kitchen_wh / total * 100, 1)},
        {"category": "Laundry room (washer/dryer/fridge/light)", "measured": True, "share_pct": round(laundry_wh / total * 100, 1)},
        {"category": "Water heater + AC", "measured": True, "share_pct": round(wh_ac_wh / total * 100, 1)},
        {"category": "Other unmetered loads", "measured": False, "share_pct": round(other_wh / total * 100, 1)},
    ]
    return {
        "available": True,
        "message": "Based on the dataset's real sub-metering channels (measured, not modeled) plus a residual 'other' category.",
        "categories": categories,
    }
