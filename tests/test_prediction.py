from src.cost_estimation import estimate_costs
from src.risk_score import calculate_risk


def test_risk_levels():
    assert calculate_risk(0.10) == (10, "LOW")
    assert calculate_risk(0.80) == (80, "CRITICAL")


def test_expected_loss():
    assert estimate_costs(0.90)["expected_failure_loss"] == 108000.0

