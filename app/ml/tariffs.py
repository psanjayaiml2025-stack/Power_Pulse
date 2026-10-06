"""
Illustrative Indian residential electricity tariff slabs.

IMPORTANT (Section 6, 29): These are approximate, illustrative slab
structures for common states, NOT official DISCOM rates - real tariffs
change often and vary by connection type/sanctioned load/subsidy status.
Every place this is used in the API, the response is labeled "estimated"
and the user can override with `custom_rate_per_kwh`.
"""

# (unit_upper_bound_kwh, rate_per_kwh) - monthly domestic slabs, illustrative only
ILLUSTRATIVE_SLABS = {
    "Tamil Nadu": [(100, 0.0), (200, 2.35), (400, 4.7), (500, 6.3), (float("inf"), 8.4)],
    "Karnataka": [(100, 4.15), (200, 5.6), (300, 7.15), (float("inf"), 8.2)],
    "Maharashtra": [(100, 5.6), (300, 9.7), (500, 13.5), (float("inf"), 15.9)],
    "Delhi": [(200, 3.0), (400, 4.5), (800, 6.5), (float("inf"), 7.0)],
    "West Bengal": [(102, 6.0), (180, 6.8), (300, 7.4), (float("inf"), 8.4)],
    "Uttar Pradesh": [(150, 5.5), (300, 6.0), (float("inf"), 6.5)],
    "Other / Custom": [(float("inf"), 7.0)],
}


def estimate_cost(monthly_kwh: float, state: str, custom_rate: float | None = None) -> dict:
    if custom_rate is not None:
        return {
            "monthly_kwh": round(monthly_kwh, 2),
            "estimated_cost": round(monthly_kwh * custom_rate, 2),
            "rate_basis": f"custom flat rate ₹{custom_rate}/kWh (user-supplied)",
            "is_estimated": True,
        }

    slabs = ILLUSTRATIVE_SLABS.get(state, ILLUSTRATIVE_SLABS["Other / Custom"])
    remaining = monthly_kwh
    prev_bound = 0
    cost = 0.0
    for upper, rate in slabs:
        slab_units = max(0, min(remaining, upper - prev_bound))
        cost += slab_units * rate
        remaining -= slab_units
        prev_bound = upper
        if remaining <= 0:
            break

    return {
        "monthly_kwh": round(monthly_kwh, 2),
        "estimated_cost": round(cost, 2),
        "rate_basis": f"illustrative {state} domestic slab structure (verify with your DISCOM)",
        "is_estimated": True,
    }


def estimate_appliance_usage(watts: float, hours_per_day: float, days_per_month: float,
                              state: str, custom_rate: float | None = None) -> dict:
    """Section 17: separate from actual meter measurements. Pure watts x
    hours x days arithmetic - never presented as a real measurement."""
    kwh_per_day = (watts * hours_per_day) / 1000.0
    kwh_per_month = kwh_per_day * days_per_month
    cost = estimate_cost(kwh_per_month, state, custom_rate)
    return {
        "label": "Appliance Usage Estimate (not a real measurement)",
        "watts": watts,
        "hours_per_day": hours_per_day,
        "days_per_month": days_per_month,
        "estimated_kwh_per_day": round(kwh_per_day, 3),
        "estimated_kwh_per_month": round(kwh_per_month, 2),
        "estimated_monthly_cost": cost["estimated_cost"],
        "rate_basis": cost["rate_basis"],
    }
