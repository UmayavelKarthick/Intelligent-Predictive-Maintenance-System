import numpy as np


def feature_contributions(model_pipeline, input_frame, top_n: int = 3) -> list[dict]:
    """Return local SHAP contributions for the fitted final tree model."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install shap to generate model-based explanations.") from exc

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    transformed = preprocessor.transform(input_frame)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    row_values = values[0]
    ranked = np.argsort(np.abs(row_values))[::-1][:top_n]
    return [
        {"feature": str(feature_names[index]), "impact": float(row_values[index])}
        for index in ranked
    ]

