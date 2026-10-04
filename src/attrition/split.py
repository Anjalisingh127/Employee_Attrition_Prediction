"""Deterministic, leakage-safe train/test preparation."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from attrition.data import ID_COLUMN, INVARIANT_COLUMNS, TARGET_COLUMN

EXCLUDED_FEATURE_COLUMNS = (ID_COLUMN, *INVARIANT_COLUMNS)
TARGET_MAPPING = {"No": 0, "Yes": 1}


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test matrices plus their binary targets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target without fitting transformations."""

    required = (TARGET_COLUMN, *EXCLUDED_FEATURE_COLUMNS)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns for modeling: {missing}")

    unknown_targets = set(df[TARGET_COLUMN].dropna().unique()) - set(TARGET_MAPPING)
    if unknown_targets or df[TARGET_COLUMN].isna().any():
        raise ValueError("Attrition must contain only non-null 'Yes'/'No' values")

    X = df.drop(columns=[TARGET_COLUMN, *EXCLUDED_FEATURE_COLUMNS]).copy()
    y = df[TARGET_COLUMN].map(TARGET_MAPPING).astype("int8").rename("attrition")

    return X, y


def create_stratified_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_seed: int = 42,
) -> DatasetSplit:
    """Create the final holdout before learned preprocessing or resampling."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
