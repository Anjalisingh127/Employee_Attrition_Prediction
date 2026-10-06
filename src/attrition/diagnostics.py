"""Baseline comparison and fold-stability diagnostics."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.modeling import BaselineResult, evaluate_all_baselines
from attrition.split import create_stratified_split

PRIMARY_METRIC = "pr_auc"
RANKING_METRICS = ("pr_auc", "recall", "roc_auc")
DEFAULT_OUTPUT_DIR = Path("reports/generated")


@dataclass(frozen=True)
class StabilitySummary:
    """Fold-level spread for one model metric."""

    mean: float
    std: float
    minimum: float
    maximum: float
    range: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ModelComparison:
    """Decision-focused summary for one baseline model."""

    rank: int
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    pr_auc_std: float
    recall_std: float
    roc_auc_std: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def metric_stability(result: BaselineResult, metric: str) -> StabilitySummary:
    """Summarize fold-to-fold spread for one metric."""

    if metric not in result.metrics:
        raise ValueError(f"unknown metric: {metric}")

    summary = result.metrics[metric]
    minimum = min(summary.folds)
    maximum = max(summary.folds)
    return StabilitySummary(
        mean=summary.mean,
        std=summary.std,
        minimum=minimum,
        maximum=maximum,
        range=maximum - minimum,
    )


def rank_baselines(results: list[BaselineResult]) -> list[ModelComparison]:
    """Rank models transparently by PR-AUC, recall, then ROC-AUC."""

    if not results:
        raise ValueError("at least one baseline result is required")

    ordered = sorted(
        results,
        key=lambda result: tuple(
            result.metrics[metric].mean for metric in RANKING_METRICS
        ),
        reverse=True,
    )

    return [
        ModelComparison(
            rank=index,
            model=result.model,
            accuracy=result.metrics["accuracy"].mean,
            precision=result.metrics["precision"].mean,
            recall=result.metrics["recall"].mean,
            f1=result.metrics["f1"].mean,
            roc_auc=result.metrics["roc_auc"].mean,
            pr_auc=result.metrics["pr_auc"].mean,
            pr_auc_std=result.metrics["pr_auc"].std,
            recall_std=result.metrics["recall"].std,
            roc_auc_std=result.metrics["roc_auc"].std,
        )
        for index, result in enumerate(ordered, start=1)
    ]


def build_diagnostics(results: list[BaselineResult]) -> dict[str, object]:
    """Build a machine-readable comparison and stability report."""

    ranking = rank_baselines(results)
    return {
        "selection_policy": {
            "primary_metric": PRIMARY_METRIC,
            "ranking_order": list(RANKING_METRICS),
            "weighted_composite_score": False,
        },
        "ranking": [item.to_dict() for item in ranking],
        "stability": {
            result.model: {
                metric: metric_stability(result, metric).to_dict()
                for metric in ("pr_auc", "recall", "roc_auc")
            }
            for result in results
        },
        "recommended_baseline": ranking[0].model,
        "interpretation": (
            "Recommendation is based on training-only cross-validation and may change "
            "after imbalance handling, tuning, threshold analysis, and final evaluation."
        ),
    }


def comparison_frame(results: list[BaselineResult]) -> pd.DataFrame:
    """Return a recruiter-readable baseline comparison table."""

    ranking = rank_baselines(results)
    return pd.DataFrame([item.to_dict() for item in ranking])


def save_diagnostics(
    results: list[BaselineResult],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write reproducible CSV and JSON diagnostics under the generated-report directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_comparison.csv"
    json_path = output_dir / "baseline_diagnostics.json"

    comparison_frame(results).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(build_diagnostics(results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return csv_path, json_path


def main() -> None:
    """Evaluate baselines, save diagnostics, and print the comparison report."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )
    results = evaluate_all_baselines(
        split.X_train,
        split.y_train,
        folds=5,
        random_seed=settings.random_seed,
    )
    csv_path, json_path = save_diagnostics(results)
    payload = {
        "evaluation": "training-only stratified cross-validation diagnostics",
        "final_holdout_rows": len(split.X_test),
        "holdout_evaluated": False,
        "artifacts": {
            "comparison_csv": str(csv_path),
            "diagnostics_json": str(json_path),
        },
        **build_diagnostics(results),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
