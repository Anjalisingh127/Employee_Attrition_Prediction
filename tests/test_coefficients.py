"""Tests for Stage 7.1 Logistic Regression coefficient analysis."""

from pathlib import Path

import numpy as np
import pytest

from attrition.coefficients import (
    _clean_feature_name,
    coefficient_table,
    fit_explainable_base_model,
    ranked_drivers,
)
from attrition.data import load_dataset
from attrition.split import create_stratified_split

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def fitted_pipeline():
    df = load_dataset(DATASET)
    split = create_stratified_split(df, random_seed=42)
    sample_X = split.X_train.iloc[:400]
    sample_y = split.y_train.loc[sample_X.index]
    return fit_explainable_base_model(sample_X, sample_y, random_seed=42)


def test_clean_feature_name_removes_transformer_prefix() -> None:
    assert _clean_feature_name("numeric__Age") == "Age"
    assert _clean_feature_name("categorical__OverTime_Yes") == "OverTime_Yes"


def test_coefficient_table_aligns_names_and_coefficients(fitted_pipeline) -> None:
    table = coefficient_table(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]

    assert len(table) == model.coef_.shape[1]
    assert table["feature"].is_unique
    assert np.isfinite(table["coefficient"]).all()
    assert np.isfinite(table["odds_ratio"]).all()


def test_coefficients_are_ranked_by_absolute_magnitude(fitted_pipeline) -> None:
    table = coefficient_table(fitted_pipeline)

    values = table["absolute_coefficient"].to_numpy()
    assert np.all(values[:-1] >= values[1:])


def test_direction_matches_coefficient_sign(fitted_pipeline) -> None:
    table = coefficient_table(fitted_pipeline)

    positive = table.loc[table["coefficient"] > 0]
    negative = table.loc[table["coefficient"] < 0]
    assert set(positive["direction"]) == {"higher_attrition_log_odds"}
    assert set(negative["direction"]) == {"lower_attrition_log_odds"}


def test_ranked_drivers_return_both_directions(fitted_pipeline) -> None:
    drivers = ranked_drivers(coefficient_table(fitted_pipeline), top_n=5)

    assert len(drivers["higher_attrition_log_odds"]) == 5
    assert len(drivers["lower_attrition_log_odds"]) == 5
    assert all(
        item["coefficient"] > 0
        for item in drivers["higher_attrition_log_odds"]
    )
    assert all(
        item["coefficient"] < 0
        for item in drivers["lower_attrition_log_odds"]
    )


def test_invalid_top_n_is_rejected(fitted_pipeline) -> None:
    with pytest.raises(ValueError, match="top_n must be at least 1"):
        ranked_drivers(coefficient_table(fitted_pipeline), top_n=0)
