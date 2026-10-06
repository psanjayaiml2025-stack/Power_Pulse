"""
PowerPulse Energy Health Score - Section 3.

An explainable 0-100 score built from four transparent, weighted
components. NOT an official or industry-certified score - always
labeled "PowerPulse Energy Health Score" and always shown with its
component breakdown so the number is never a black box.
"""


def compute_health_score(pct_change_vs_baseline, anomalies_count, total_days,
                          peak_period_share_pct, future_vs_baseline_pct):
    components = {}

    dev = abs(pct_change_vs_baseline) if pct_change_vs_baseline is not None else 0
    components["baseline_alignment"] = round(max(0, 40 - dev * 0.8), 1)

    rate_pct = (anomalies_count / total_days * 100) if total_days else 0
    components["anomaly_control"] = round(max(0, 30 - rate_pct * 3), 1)

    peak_component = 15.0
    if peak_period_share_pct is not None:
        excess = max(0, peak_period_share_pct - 20)
        peak_component = max(0, 15 - excess * 0.5)
    components["peak_balance"] = round(peak_component, 1)

    trend_component = 15.0
    if future_vs_baseline_pct is not None:
        trend_component = max(0, 15 - max(0, future_vs_baseline_pct) * 0.3)
    components["trend_outlook"] = round(trend_component, 1)

    total = max(0, min(100, round(sum(components.values()))))

    if total >= 80:
        label = "Excellent"
    elif total >= 60:
        label = "Good"
    elif total >= 40:
        label = "Needs Attention"
    else:
        label = "Poor"

    return {
        "score": total,
        "label": label,
        "components": components,
        "component_max": {"baseline_alignment": 40, "anomaly_control": 30, "peak_balance": 15, "trend_outlook": 15},
        "methodology": (
            "PowerPulse Energy Health Score (0-100, not an official/industry-certified "
            "score): 40 pts for staying close to your baseline, 30 pts for low anomaly "
            "frequency, 15 pts for balanced (non-spiky) peak-hour usage, 15 pts for a "
            "flat-or-improving forecast trend."
        ),
    }
