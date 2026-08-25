# Intelligent Predictive Maintenance System

An end-to-end college project built from the AI4I 2020 Predictive Maintenance dataset. It predicts machine-failure probability, assigns a transparent risk level, predicts the likely failure type when failure is predicted, produces model-based SHAP explanations, and compares expected failure loss with preventive-maintenance cost.

## Project layout

```text
data/raw/                 Original source data used by the pipeline
data/processed/           Reproducible cleaned data created during training
src/                      Data, training, prediction and business-logic modules
models/                   Saved model artifacts (created by training)
outputs/metrics/          Model comparison and evaluation results
app/app.py                Streamlit demonstration dashboard
tests/                    Small logic tests
```

## Important modelling decisions

- The binary target is `Machine failure`; the five failure-indicator columns are **never** used as input features because they would leak the answer.
- The data is imbalanced (339 failures among 10,000 records). Models use class weights and are selected by F1 after recall-oriented threshold tuning (minimum recall target: 0.80), not accuracy alone.
- Failure type is a separate classifier trained only on real failures. The dataset has 24 multi-cause rows; these are excluded from its single-label training set and documented in `src/data_cleaning.py`.
- â‚¹8,000 preventive maintenance, â‚¹120,000 potential failure, and â‚¹2,000 unnecessary-maintenance costs are example project assumptions. Update `CostAssumptions` before presenting a different business context.

## Run locally

Create and activate a Python 3.12 virtual environment if one does not already exist:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train the models from the project root:

```powershell
python -m src.train_failure_model
python -m src.train_failure_type_model
```

Start the dashboard:

```powershell
streamlit run app/app.py
```

Run the small automated checks:

```powershell
python -m pytest
```

## Outputs created after training

- `models/failure_model.joblib` â€” selected binary classifier, threshold, and metrics.
- `models/failure_type_model.joblib` â€” failure-type classifier.
- `outputs/metrics/failure_model_comparison.csv` â€” Logistic Regression, Decision Tree, Random Forest, XGBoost and Neural Network comparison.
- `outputs/metrics/failure_model_selected.json` â€” selected model metrics and confusion matrix.
- `outputs/metrics/failure_type_model_report.json` â€” failure-type classification report.

