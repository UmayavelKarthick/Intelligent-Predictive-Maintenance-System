"""Train a separate classifier only for machines that actually failed."""
import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, METRICS_DIR, MODELS_DIR
from src.data_cleaning import validate_and_clean_data
from src.data_loader import load_raw_data
from src.preprocessing import make_pipeline


def train_failure_type_model() -> dict:
    data = validate_and_clean_data(load_raw_data())
    failures = data[data["Machine failure"] == 1].copy()
    # Exclude 24 multi-cause rows: a conventional multi-class model requires one true label.
    failures = failures[~failures["failure_type_is_multicause"]]
    x_train, x_test, y_train, y_test = train_test_split(
        failures[FEATURE_COLUMNS], failures["failure_type"], test_size=0.25,
        random_state=42, stratify=failures["failure_type"]
    )
    pipeline = make_pipeline(RandomForestClassifier(
        n_estimators=400, class_weight="balanced", min_samples_leaf=2, random_state=42, n_jobs=-1
    ))
    pipeline.fit(x_train, y_train)
    report = classification_report(y_test, pipeline.predict(x_test), output_dict=True, zero_division=0)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "classes": list(pipeline.classes_)}, MODELS_DIR / "failure_type_model.joblib")
    (METRICS_DIR / "failure_type_model_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = train_failure_type_model()
    print(json.dumps({"accuracy": report.get("accuracy")}, indent=2))

