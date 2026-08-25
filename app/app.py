from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.explainability import feature_contributions
from src.prediction import load_models, predict_machine

st.set_page_config(page_title="Predictive Maintenance", page_icon="âš™ï¸", layout="wide")
st.title("Intelligent Predictive Maintenance System")
st.caption("AI4I 2020 dataset â€¢ Cost figures are configurable project assumptions, not company data.")

with st.sidebar:
    st.header("Machine sensor input")
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)
    air_temperature = st.number_input("Air Temperature (K)", 290.0, 315.0, 300.0, 0.1)
    process_temperature = st.number_input("Process Temperature (K)", 300.0, 325.0, 310.0, 0.1)
    rotational_speed = st.number_input("Rotational Speed (rpm)", 100.0, 3000.0, 1500.0, 1.0)
    torque = st.number_input("Torque (Nm)", 1.0, 100.0, 40.0, 0.1)
    tool_wear = st.number_input("Tool Wear (min)", 0.0, 300.0, 100.0, 1.0)
    analyse = st.button("Analyse Machine", type="primary")

if analyse:
    sensor_data = {
        "Type": machine_type, "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed, "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    try:
        result = predict_machine(sensor_data)
        status = "YES" if result["machine_failure"] else "NO"
        st.subheader("Machine Health Result")
        left, middle, right = st.columns(3)
        left.metric("Machine Failure", status)
        middle.metric("Failure Probability", f"{result['failure_probability']:.1%}")
        right.metric("Risk Score", f"{result['risk_score']} / 100", result["risk_level"])
        st.write(f"**Likely Failure Type:** {result['likely_failure_type']}")
        st.write(f"**Maintenance Recommendation:** {result['maintenance_recommendation']}")
        costs = result["cost_estimate"]
        st.subheader("Business Decision (example assumptions)")
        a, b, c = st.columns(3)
        a.metric("Potential Failure Cost", f"â‚¹{costs['potential_failure_cost']:,.0f}")
        b.metric("Expected Failure Loss", f"â‚¹{costs['expected_failure_loss']:,.0f}")
        c.metric("Preventive Maintenance", f"â‚¹{costs['preventive_maintenance_cost']:,.0f}")
        st.info(costs["business_decision"])
        try:
            failure_artifact, _ = load_models()
            factors = feature_contributions(failure_artifact["pipeline"], pd.DataFrame([sensor_data]))
            st.subheader("Main Contributing Factors (SHAP)")
            for factor in factors:
                direction = "increased" if factor["impact"] > 0 else "reduced"
                st.write(f"â€¢ {factor['feature']} {direction} predicted failure risk.")
        except RuntimeError as error:
            st.warning(str(error))
    except FileNotFoundError as error:
        st.error(f"{error} Training command: `python -m src.train_failure_model` then `python -m src.train_failure_type_model`")
    except Exception as error:
        st.error(f"Unable to analyse this input: {error}")
else:
    st.info("Enter sensor values in the sidebar and select Analyse Machine.")

