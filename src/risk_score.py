def calculate_risk(probability: float) -> tuple[int, str]:
    """Map probability to a transparent 0-100 project risk score."""
    score = round(max(0.0, min(1.0, probability)) * 100)
    if score < 20:
        level = "LOW"
    elif score < 50:
        level = "MEDIUM"
    elif score < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"
    return score, level

