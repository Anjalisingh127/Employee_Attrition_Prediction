"""Tests for deterministic business feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attrition.data import load_dataset
from attrition.features import ENGINEERED_FEATURES, add_business_features
from attrition.split import prepare_features_and_target

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def model_features() -> pd.DataFrame:
    df = load_dataset(DATASET)
    X, _ = prepare_features_and_target(df)
    return X


def test_business_features_are_added_without_mutating_input(
    model_features: pd.DataFrame,
) -> None:
    original = model_features.copy(deep=True)
    engineered = add_business_features(model_features)

    pd.testing.assert_frame_equal(model_features, original)
    assert engineered.shape == (1470, 35)
    assert set(ENGINEERED_FEATURES).issubset(engineered.columns)


def test_engineered_features_are_finite(model_features: pd.DataFrame) -> None:
    engineered = add_business_features(model_features)

    values = engineered[list(ENGINEERED_FEATURES)].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_engineered_ratios_match_known_row(model_features: pd.DataFrame) -> None:
    engineered = add_business_features(model_features)
    row = engineered.iloc[0]

    assert row["IncomePerJobLevel"] == pytest.approx(5993 / 2)
    assert row["CompanyTenureRatio"] == pytest.approx(6 / 8)
    assert row["RoleTenureRatio"] == pytest.approx(4 / 6)
    assert row["PromotionWaitRatio"] == pytest.approx(0.0)
    assert row["EarlyCareer"] == 0


def test_zero_denominators_do_not_create_nan_or_infinity(
    model_features: pd.DataFrame,
) -> None:
    sample = model_features.iloc[[0]].copy()
    sample["TotalWorkingYears"] = 0
    sample["YearsAtCompany"] = 0

    engineered = add_business_features(sample)

    assert engineered.iloc[0]["CompanyTenureRatio"] == 0.0
    assert engineered.iloc[0]["RoleTenureRatio"] == 0.0
    assert engineered.iloc[0]["PromotionWaitRatio"] == 0.0


def test_missing_required_column_is_rejected(model_features: pd.DataFrame) -> None:
    broken = model_features.drop(columns=["MonthlyIncome"])

    with pytest.raises(ValueError, match="missing columns required"):
        add_business_features(broken)
