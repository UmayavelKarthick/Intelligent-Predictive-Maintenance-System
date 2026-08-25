import pandas as pd

from src.config import RAW_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original AI4I CSV and fail early if it is unavailable."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

