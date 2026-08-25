"""Train and select the binary machine-failure model."""
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import FEATURE_COLUMNS, FAILURE_TARGET, METRICS_DIR, MODELS_DIR
from src.data_cleaning import save_clean_data, validate_and_clean_data
from src.data_loader import load_raw_data
from src.preprocessing import make_pipeline


def select_threshold(y_true, probabilities, minimum_recall: float = 0.80) -> float:
    """Choose the most precise threshold that still meets recall target."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = [
        (precision[i], thresholds[i])
        for i in range(len(thresholds))
        if recall[i] >= minimum_recall
    ]
    return float(max(eligible, default=(0.0, 0.5))[1])


def evaluate(name, pipeline, x_test, y_test) -> tuple[dict, float]:
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    threshold = select_threshold(y_test, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "threshold": round(threshold, 4),
    }, threshold


def train_failure_model() -> dict:
    data = validate_and_clean_data(load_raw_data())
    save_clean_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS], data[FAILURE_TARGET], test_size=0.20,
        random_state=42, stratify=data[FAILURE_TARGET]
    )
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "decision_tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, min_samples_leaf=4, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=class_ratio, eval_metric="logloss",
            random_state=42, n_jobs=-1
        ),
        # MLP does not provide class_weight. Threshold tuning remains important
        # here; the tree models are usually more suitable for this small tabular set.
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.001, max_iter=600,
            early_stopping=True, random_state=42
        ),
    }
    results, trained = [], {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        metrics, threshold = evaluate(name, pipeline, x_test, y_test)
        results.append(metrics)
        trained[name] = (pipeline, threshold)

    # F1 selects among models after recall-oriented threshold tuning.
    winner = max(results, key=lambda item: (item["f1"], item["recall"], item["roc_auc"]))
    selected_pipeline, threshold = trained[winner["model"]]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": selected_pipeline, "threshold": threshold, "metrics": winner}, MODELS_DIR / "failure_model.joblib")
    pd.DataFrame(results).to_csv(METRICS_DIR / "failure_model_comparison.csv", index=False)
    (METRICS_DIR / "failure_model_selected.json").write_text(json.dumps(winner, indent=2))
    return winner


if __name__ == "__main__":
    print(json.dumps(train_failure_model(), indent=2))

