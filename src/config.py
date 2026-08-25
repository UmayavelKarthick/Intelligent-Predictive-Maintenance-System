from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned_data.csv"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FAILURE_TARGET = "Machine failure"
FAILURE_INDICATORS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_TYPE_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

