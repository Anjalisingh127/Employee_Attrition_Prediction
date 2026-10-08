"""Tests for Stage 7.4 local SHAP explainability."""

import numpy as np
import pandas as pd
import pytest
import shap

from attrition.shap_local import (
    local_contribution_table,
    top_local_contributions,
)


def _example_explanation() -> shap.Explanation:
    return shap.Explanation(
        values=np.array(
            [
                [0.4, -0.3, 0.1],
                [-0.2, 0.5, -0.1],
            ]
        ),
        base_values=np.array([-1.0, -1.0]),
        data=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
        feature_names=["A", "B", "C"],
    )


def test_local_table_ranks_absolute_contribution() -> None:
    table = local_contribution_table(_example_explanation(), 0)

    assert table.iloc[0]["feature"] == "A"
    assert table.iloc[0]["absolute_shap"] == pytest.approx(0.4)


def test_local_table_preserves_direction() -> None:
    table = local_contribution_table(_example_explanation(), 0)

    feature_a = table.loc[table["feature"] == "A"].iloc[0]
    feature_b = table.loc[table["feature"] == "B"].iloc[0]
    assert feature_a["direction"] == "increases_attrition_score"
    assert feature_b["direction"] == "decreases_attrition_score"


def test_local_table_preserves_transformed_value() -> None:
    table = local_contribution_table(_example_explanation(), 0)

    feature_c = table.loc[table["feature"] == "C"].iloc[0]
    assert feature_c["transformed_value"] == pytest.approx(3.0)


def test_invalid_row_position_is_rejected() -> None:
    with pytest.raises(IndexError, match="outside the explained dataset"):
        local_contribution_table(_example_explanation(), 5)


def test_top_local_contributions_split_directions() -> None:
    table = local_contribution_table(_example_explanation(), 0)
    result = top_local_contributions(table, top_n=2)

    assert [item["feature"] for item in result["risk_increasing"]] == ["A", "C"]
    assert [item["feature"] for item in result["risk_reducing"]] == ["B"]


def test_invalid_top_n_is_rejected() -> None:
    table = pd.DataFrame(
        {
            "feature": ["A"],
            "transformed_value": [1.0],
            "shap_value": [0.1],
            "absolute_shap": [0.1],
            "direction": ["increases_attrition_score"],
        }
    )

    with pytest.raises(ValueError, match="top_n must be at least 1"):
        top_local_contributions(table, top_n=0)
