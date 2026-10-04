"""Tests for leakage-safe preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from attrition.data import load_dataset
from attrition.preprocessing import build_preprocessor, infer_feature_groups
from attrition.split import create_stratified_split

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def split():
    df = load_dataset(DATASET)
    return create_stratified_split(df, random_seed=42)


def test_feature_groups_cover_engineered_schema(split) -> None:
    groups = infer_feature_groups(split.X_train)

    assert len(groups.numeric) == 28
    assert len(groups.categorical) == 7
    assert set(groups.numeric).isdisjoint(groups.categorical)
    assert len(groups.numeric) + len(groups.categorical) == 35
    assert "Department" in groups.categorical
    assert "IncomePerJobLevel" in groups.numeric


def test_preprocessor_is_unfitted_when_built(split) -> None:
    preprocessor = build_preprocessor(split.X_train)

    with pytest.raises(NotFittedError):
        preprocessor.transform(split.X_test)


def test_preprocessor_is_sklearn_cloneable(split) -> None:
    preprocessor = build_preprocessor(split.X_train)

    cloned = clone(preprocessor)

    assert cloned is not preprocessor
    assert cloned.steps[0][0] == "business_features"


def test_fit_transform_produces_finite_matrix_without_mutating_training_data(split) -> None:
    original = split.X_train.copy(deep=True)
    preprocessor = build_preprocessor(split.X_train)

    transformed = preprocessor.fit_transform(split.X_train, split.y_train)

    pd.testing.assert_frame_equal(split.X_train, original)
    assert transformed.shape[0] == len(split.X_train)
    assert transformed.shape[1] > split.X_train.shape[1]
    assert np.isfinite(transformed).all()


def test_fitted_preprocessor_handles_unseen_category(split) -> None:
    preprocessor = build_preprocessor(split.X_train)
    preprocessor.fit(split.X_train, split.y_train)

    sample = split.X_test.iloc[[0]].copy()
    sample["Department"] = "Future Department"
    transformed = preprocessor.transform(sample)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_train_and_test_have_same_transformed_schema(split) -> None:
    preprocessor = build_preprocessor(split.X_train)
    train_matrix = preprocessor.fit_transform(split.X_train, split.y_train)
    test_matrix = preprocessor.transform(split.X_test)

    assert train_matrix.shape[1] == test_matrix.shape[1]
    names = preprocessor.get_feature_names_out()
    assert len(names) == train_matrix.shape[1]
