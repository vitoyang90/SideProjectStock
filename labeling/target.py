import pandas as pd
import numpy as np


def add_5day_direction_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add the future five-trading-day return and binary direction label."""
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    labeled_df = df.copy()
    labeled_df["future_close_5d"] = labeled_df["close"].shift(-5)
    labeled_df["return_5d"] = (
        labeled_df["future_close_5d"] - labeled_df["close"]
    ) / labeled_df["close"]

    # The last five rows have no known future close and cannot be labeled.
    labeled_df = labeled_df.dropna(subset=["future_close_5d"]).copy()
    labeled_df["label"] = (labeled_df["return_5d"] > 0).astype("int8")

    return labeled_df


def add_5day_three_class_label(
    df: pd.DataFrame,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """Add down/neutral/up labels using a symmetric return threshold.

    Labels: 0 = down, 1 = neutral, 2 = up.
    """
    if "return_5d" not in df.columns:
        raise ValueError("DataFrame must contain a 'return_5d' column")
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")

    labeled_df = df.copy()
    labeled_df["target_3class"] = np.select(
        [
            labeled_df["return_5d"] < -threshold,
            labeled_df["return_5d"] > threshold,
        ],
        [0, 2],
        default=1,
    ).astype("int8")

    return labeled_df
