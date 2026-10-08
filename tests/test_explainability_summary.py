"""Tests for Stage 7.5 explainability synthesis."""

import pandas as pd
import pytest

from attrition.explainability_summary import (
    aggregate_transformed_importance,
    consensus_table,
    feature_family,
    rank_table,
    render_recruiter_report,
)


def test_feature_family_maps_one_hot_terms() -> None:
    original = ["OverTime", "JobRole", "Age"]

    assert feature_family("OverTime_Yes", original) == "OverTime"
    assert feature_family("JobRole_Laboratory Technician", original) == "JobRole"
    assert feature_family("Age", original) == "Age"


def test_feature_family_preserves_engineered_feature() -> None:
    assert feature_family("RoleTenureRatio", ["Age", "JobRole"]) == "RoleTenureRatio"


def test_aggregate_transformed_importance_sums_categories() -> None:
    table = pd.DataFrame(
        {
            "feature": ["OverTime_No", "OverTime_Yes", "Age"],
            "absolute_coefficient": [1.0, 0.5, 0.3],
        }
    )

    result = aggregate_transformed_importance(
        table,
        value_column="absolute_coefficient",
        original_features=["OverTime", "Age"],
    )

    overtime = result.loc[result["feature_family"] == "OverTime"].iloc[0]
    assert overtime["absolute_coefficient"] == pytest.approx(1.5)


def test_rank_table_rejects_invalid_top_n() -> None:
    table = pd.DataFrame({"feature_family": ["A"], "importance": [0.2]})

    with pytest.raises(ValueError, match="top_n must be at least 1"):
        rank_table(table, value_column="importance", method="test", top_n=0)


def test_consensus_counts_independent_methods() -> None:
    coefficient = pd.DataFrame(
        {
            "feature": ["A", "B"],
            "absolute_coefficient": [0.9, 0.8],
        }
    )
    permutation = pd.DataFrame(
        {
            "feature": ["A", "C"],
            "average_precision_importance_mean": [0.3, 0.2],
        }
    )
    shap = pd.DataFrame(
        {
            "feature": ["A", "B"],
            "mean_abs_shap": [0.7, 0.6],
        }
    )

    result = consensus_table(
        coefficient,
        permutation,
        shap,
        ["A", "B", "C"],
        top_n=2,
    )

    feature_a = result.loc[result["feature_family"] == "A"].iloc[0]
    assert feature_a["methods"] == 3
    assert feature_a["mean_rank"] == pytest.approx(1.0)


def test_recruiter_report_contains_guardrails() -> None:
    summary = {
        "cross_method_features": [
            {
                "feature_family": "OverTime",
                "methods": 3,
                "method_list": "coefficient, permutation, shap",
                "mean_rank": 1.3,
            }
        ],
        "directional_terms": {
            "higher_attrition_log_odds": [
                {"feature": "OverTime_Yes", "coefficient": 0.5, "odds_ratio": 1.65}
            ],
            "lower_attrition_log_odds": [
                {"feature": "OverTime_No", "coefficient": -0.8, "odds_ratio": 0.45}
            ],
        },
        "permutation_validation": {
            "average_precision": {"mean": 0.67, "std": 0.05},
            "roc_auc": {"mean": 0.84, "std": 0.03},
        },
    }

    report = render_recruiter_report(summary)

    assert "not causal" in report
    assert "final 294-row holdout is not reused" in report
    assert "OverTime" in report
