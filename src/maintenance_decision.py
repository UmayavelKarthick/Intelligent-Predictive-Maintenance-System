RECOMMENDATIONS = {
    "LOW": "Continue monitoring during normal operation.",
    "MEDIUM": "Schedule preventive maintenance in the next planned window.",
    "HIGH": "Prepare maintenance and inspect the machine as soon as possible.",
    "CRITICAL": "Immediate maintenance recommended; avoid continued operation.",
}


def recommend_maintenance(risk_level: str) -> str:
    return RECOMMENDATIONS[risk_level]

