from dataclasses import dataclass


@dataclass(frozen=True)
class CostAssumptions:
    preventive_maintenance_cost: float = 8000.0
    potential_failure_cost: float = 120000.0
    unnecessary_maintenance_cost: float = 2000.0


def estimate_costs(probability: float, costs: CostAssumptions = CostAssumptions()) -> dict:
    expected_failure_loss = probability * costs.potential_failure_cost
    action_is_economic = expected_failure_loss > costs.preventive_maintenance_cost
    return {
        "potential_failure_cost": costs.potential_failure_cost,
        "expected_failure_loss": expected_failure_loss,
        "preventive_maintenance_cost": costs.preventive_maintenance_cost,
        "action_is_economic": action_is_economic,
        "business_decision": (
            "Perform preventive maintenance: expected loss exceeds preventive cost."
            if action_is_economic
            else "Continue monitoring: expected loss is below preventive cost."
        ),
    }

