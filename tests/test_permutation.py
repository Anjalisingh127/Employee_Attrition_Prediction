"""Tests for Stage 7.2 permutation importance."""

import numpy as np
import pandas as pd
import pytest

from attrition.permutation import (
    PRIMARY_IMPORTANCE_METRIC,
    SECONDARY_IMPORTANCE_METRIC,
    _importance_table,
    top_features,
)


def test_importance_table_aggregates_folds_and_repeats() -> None:
    metric_importances = {
        "average_precision": [
            np.array([[0.10, 0.20], [0.01, 0.02]]),
            np.array([[0.30, 0.40], [0.03, 0.04]]),
        ],
        "roc_auc": [
            np.array([[0.05, 0.10], [0.00, 0.01]]),
            np.array([[0.15, 0.20], [0.01, 0.02]]),
        ],
    }

    table = _importance_table(["FeatureA", "FeatureB"], metric_importances)

    assert table.iloc[0]["feature"] == "FeatureA"
    assert table.iloc[0]["average_precision_importance_mean"] == pytest.approx(0.25)
    assert table.iloc[1]["average_precision_importance_mean"] == pytest.approx(0.025)


def test_importance_table_preserves_negative_values() -> None:
    metric_importances = {
        "average_precision": [np.array([[0.10, 0.20], [-0.01, -0.02]])],
        "roc_auc": [np.array([[0.05, 0.10], [-0.01, 0.00]])],
    }

    table = _importance_table(["Useful", "Noisy"], metric_importances)

    noisy = table.loc[table["feature"] == "Noisy"].iloc[0]
    assert noisy["average_precision_importance_mean"] < 0


def test_top_features_uses_primary_metric_order() -> None:
    table = pd.DataFrame(
        {
            "feature": ["A", "B", "C"],
            "average_precision_importance_mean": [0.3, 0.2, 0.1],
            "average_precision_importance_std": [0.01, 0.02, 0.03],
            "roc_auc_importance_mean": [0.1, 0.3, 0.2],
            "roc_auc_importance_std": [0.01, 0.01, 0.01],
        }
    )

    result = top_features(table, top_n=2)

    assert [item["feature"] for item in result] == ["A", "B"]


def test_invalid_top_n_is_rejected() -> None:
    table = pd.DataFrame(
        {
            "feature": ["A"],
            "average_precision_importance_mean": [0.1],
            "average_precision_importance_std": [0.01],
            "roc_auc_importance_mean": [0.1],
            "roc_auc_importance_std": [0.01],
        }
    )

    with pytest.raises(ValueError, match="top_n must be at least 1"):
        top_features(table, top_n=0)


def test_metric_names_are_explicit() -> None:
    assert PRIMARY_IMPORTANCE_METRIC == "average_precision"
    assert SECONDARY_IMPORTANCE_METRIC == "roc_auc"
