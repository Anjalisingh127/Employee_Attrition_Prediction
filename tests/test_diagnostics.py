"""Tests for Stage 4.2 baseline comparison and diagnostics."""

from pathlib import Path

import pandas as pd
import pytest

from attrition.diagnostics import (
    build_diagnostics,
    comparison_frame,
    metric_stability,
    rank_baselines,
    save_diagnostics,
)
from attrition.modeling import BaselineResult, MetricSummary


def _result(
    model: str,
    *,
    pr_auc: float,
    recall: float,
    roc_auc: float,
) -> BaselineResult:
    metrics = {
        "accuracy": MetricSummary(0.85, 0.01, (0.84, 0.85, 0.86)),
        "precision": MetricSummary(0.70, 0.02, (0.68, 0.70, 0.72)),
        "recall": MetricSummary(recall, 0.03, (recall - 0.03, recall, recall + 0.03)),
        "f1": MetricSummary(0.60, 0.02, (0.58, 0.60, 0.62)),
        "roc_auc": MetricSummary(
            roc_auc,
            0.02,
            (roc_auc - 0.02, roc_auc, roc_auc + 0.02),
        ),
        "pr_auc": MetricSummary(
            pr_auc,
            0.04,
            (pr_auc - 0.04, pr_auc, pr_auc + 0.04),
        ),
    }
    return BaselineResult(
        model=model,
        folds=3,
        training_rows=1176,
        positive_rows=190,
        metrics=metrics,
    )


@pytest.fixture
def sample_results() -> list[BaselineResult]:
    return [
        _result("random_forest", pr_auc=0.55, recall=0.21, roc_auc=0.81),
        _result("logistic_regression", pr_auc=0.68, recall=0.49, roc_auc=0.84),
        _result("dummy", pr_auc=0.16, recall=0.00, roc_auc=0.50),
    ]


def test_rank_baselines_prioritizes_pr_auc(sample_results) -> None:
    ranked = rank_baselines(sample_results)

    assert [item.model for item in ranked] == [
        "logistic_regression",
        "random_forest",
        "dummy",
    ]
    assert [item.rank for item in ranked] == [1, 2, 3]


def test_ranking_uses_recall_as_tie_breaker() -> None:
    lower_recall = _result("a", pr_auc=0.60, recall=0.30, roc_auc=0.90)
    higher_recall = _result("b", pr_auc=0.60, recall=0.50, roc_auc=0.70)

    ranked = rank_baselines([lower_recall, higher_recall])

    assert ranked[0].model == "b"


def test_metric_stability_reports_observed_fold_range() -> None:
    result = _result("model", pr_auc=0.60, recall=0.40, roc_auc=0.80)

    stability = metric_stability(result, "pr_auc")

    assert stability.minimum == pytest.approx(0.56)
    assert stability.maximum == pytest.approx(0.64)
    assert stability.range == pytest.approx(0.08)


def test_diagnostics_are_transparent_about_selection_policy(sample_results) -> None:
    diagnostics = build_diagnostics(sample_results)

    assert diagnostics["recommended_baseline"] == "logistic_regression"
    assert diagnostics["selection_policy"]["primary_metric"] == "pr_auc"
    assert diagnostics["selection_policy"]["weighted_composite_score"] is False


def test_comparison_frame_and_artifacts_are_reproducible(
    sample_results,
    tmp_path: Path,
) -> None:
    frame = comparison_frame(sample_results)
    csv_path, json_path = save_diagnostics(sample_results, tmp_path)

    assert isinstance(frame, pd.DataFrame)
    assert frame["model"].tolist()[0] == "logistic_regression"
    assert csv_path.is_file()
    assert json_path.is_file()
    assert "recommended_baseline" in json_path.read_text(encoding="utf-8")


def test_empty_results_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one baseline"):
        rank_baselines([])
