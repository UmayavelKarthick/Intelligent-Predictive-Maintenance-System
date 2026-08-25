import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, MODELS_DIR
from src.cost_estimation import estimate_costs
from src.maintenance_decision import recommend_maintenance
from src.risk_score import calculate_risk


def load_models():
    failure_path = MODELS_DIR / "failure_model.joblib"
    type_path = MODELS_DIR / "failure_type_model.joblib"
    if not failure_path.exists() or not type_path.exists():
        raise FileNotFoundError("Models are missing. Run both training scripts first.")
    return joblib.load(failure_path), joblib.load(type_path)


def predict_machine(sensor_data: dict) -> dict:
    missing = set(FEATURE_COLUMNS).difference(sensor_data)
    if missing:
        raise ValueError(f"Missing input fields: {sorted(missing)}")
    frame = pd.DataFrame([{key: sensor_data[key] for key in FEATURE_COLUMNS}])
    failure_artifact, type_artifact = load_models()
    probability = float(failure_artifact["pipeline"].predict_proba(frame)[0, 1])
    predicted_failure = probability >= failure_artifact["threshold"]
    risk_score, risk_level = calculate_risk(probability)
    likely_type = "No failure predicted"
    type_confidence = None
    if predicted_failure:
        type_probabilities = type_artifact["pipeline"].predict_proba(frame)[0]
        best_index = type_probabilities.argmax()
        likely_type = type_artifact["classes"][best_index]
        type_confidence = float(type_probabilities[best_index])
    return {
        "machine_failure": bool(predicted_failure),
        "failure_probability": probability,
        "decision_threshold": failure_artifact["threshold"],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "likely_failure_type": likely_type,
        "failure_type_confidence": type_confidence,
        "maintenance_recommendation": recommend_maintenance(risk_level),
        "cost_estimate": estimate_costs(probability),
    }

