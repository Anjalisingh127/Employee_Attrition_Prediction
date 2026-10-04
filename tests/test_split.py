"""Tests for deterministic, leakage-safe dataset splitting."""

from pathlib import Path

import pandas as pd
import pytest

from attrition.data import load_dataset
from attrition.split import (
    EXCLUDED_FEATURE_COLUMNS,
    create_stratified_split,
    prepare_features_and_target,
)

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def source_df() -> pd.DataFrame:
    return load_dataset(DATASET)


def test_feature_contract_excludes_target_id_and_constants(source_df: pd.DataFrame) -> None:
    X, y = prepare_features_and_target(source_df)

    assert X.shape == (1470, 30)
    assert "Attrition" not in X.columns
    assert all(column not in X.columns for column in EXCLUDED_FEATURE_COLUMNS)
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 237


def test_stratified_split_has_expected_sizes_and_class_counts(source_df: pd.DataFrame) -> None:
    split = create_stratified_split(source_df, test_size=0.20, random_seed=42)

    assert split.X_train.shape == (1176, 30)
    assert split.X_test.shape == (294, 30)
    assert len(split.y_train) == 1176
    assert len(split.y_test) == 294
    assert int(split.y_train.sum()) == 190
    assert int(split.y_test.sum()) == 47


def test_split_is_deterministic(source_df: pd.DataFrame) -> None:
    first = create_stratified_split(source_df, random_seed=42)
    second = create_stratified_split(source_df, random_seed=42)

    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()


def test_train_and_test_indices_do_not_overlap(source_df: pd.DataFrame) -> None:
    split = create_stratified_split(source_df)

    assert set(split.X_train.index).isdisjoint(split.X_test.index)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_invalid_test_size_is_rejected(source_df: pd.DataFrame, test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        create_stratified_split(source_df, test_size=test_size)
