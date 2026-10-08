"""Tests for Stage 7.3 global SHAP explainability."""

import numpy as np
import pandas as pd
import pytest
import shap

from attrition.shap_global import global_importance_table, top_global_features


def _example_explanation() -> shap.Explanation:
    return shap.Explanation(
        values=np.array(
            [
                [0.5, -0.1, 0.0],
                [0.3, -0.2, 0.1],
                [-0.4, -0.3, 0.0],
            ]
        ),
        base_values=np.zeros(3),
        data=np.ones((3, 3)),
        feature_names=["A", "B", "C"],
    )


def test_global_importance_table_ranks_mean_absolute_shap() -> None:
    table = global_importance_table(_example_explanation())
    assert table.iloc[0]["feature"] == "A"
    assert table.iloc[0]["mean_abs_shap"] == pytest.approx(0.4)


def test_global_importance_table_preserves_signed_direction() -> None:
    table = global_importance_table(_example_explanation())
    feature_b = table.loc[table["feature"] == "B"].iloc[0]
    assert feature_b["mean_signed_shap"] < 0
    assert feature_b["negative_share"] == pytest.approx(1.0)


def test_global_importance_shares_are_bounded() -> None:
    table = global_importance_table(_example_explanation())
    assert table["positive_share"].between(0.0, 1.0).all()
    assert table["negative_share"].between(0.0, 1.0).all()


def test_top_global_features_uses_ranked_table() -> None:
    table = global_importance_table(_example_explanation())
    result = top_global_features(table, top_n=2)
    assert [item["feature"] for item in result] == ["A", "B"]


def test_invalid_top_n_is_rejected() -> None:
    table = pd.DataFrame(
        {
            "feature": ["A"],
            "mean_abs_shap": [0.1],
            "mean_signed_shap": [0.1],
            "positive_share": [1.0],
            "negative_share": [0.0],
        }
    )
    with pytest.raises(ValueError, match="top_n must be at least 1"):
        top_global_features(table, top_n=0)


def test_non_2d_explanation_is_rejected() -> None:
    explanation = shap.Explanation(
        values=np.array([0.1, 0.2]),
        base_values=0.0,
        feature_names=["A", "B"],
    )
    with pytest.raises(ValueError, match="2D explanation"):
        global_importance_table(explanation)
