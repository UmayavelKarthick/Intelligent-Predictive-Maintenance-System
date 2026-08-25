import pandas as pd

from src.config import (
    FAILURE_INDICATORS,
    FAILURE_TARGET,
    FAILURE_TYPE_NAMES,
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
)


def validate_and_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields, remove duplicate records, and create type labels.

    A failed machine can have multiple failure indicators.  For a single-label
    demonstration model we retain the first active indicator in the documented
    order TWF, HDF, PWF, OSF, RNF and mark such rows as multi-cause.
    """
    required = set(FEATURE_COLUMNS + [FAILURE_TARGET] + FAILURE_INDICATORS)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    cleaned = data.drop_duplicates().copy()
    if cleaned[FEATURE_COLUMNS + [FAILURE_TARGET]].isna().any().any():
        raise ValueError("Required model fields contain missing values.")

    active_counts = cleaned[FAILURE_INDICATORS].sum(axis=1)
    cleaned["failure_type_is_multicause"] = active_counts.gt(1)
    cleaned["failure_type_code"] = "No Failure"
    for indicator in FAILURE_INDICATORS:
        cleaned.loc[
            (cleaned["failure_type_code"] == "No Failure") & (cleaned[indicator] == 1),
            "failure_type_code",
        ] = indicator
    cleaned["failure_type"] = cleaned["failure_type_code"].map(FAILURE_TYPE_NAMES).fillna("No Failure")
    return cleaned


def save_clean_data(data: pd.DataFrame, path=PROCESSED_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)

