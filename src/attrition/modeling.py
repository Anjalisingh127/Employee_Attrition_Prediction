"""Leakage-safe baseline modeling with stratified cross-validation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.preprocessing import build_preprocessor
from attrition.split import create_stratified_split

BASELINE_MODEL_NAMES = ("dummy", "logistic_regression", "random_forest")
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


@dataclass(frozen=True)
class MetricSummary:
    """Cross-validation values and aggregate statistics for one metric."""

    mean: float
    std: float
    folds: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineResult:
    """Cross-validation result for one baseline estimator."""

    model: str
    folds: int
    training_rows: int
    positive_rows: int
    metrics: dict[str, MetricSummary]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "folds": self.folds,
            "training_rows": self.training_rows,
            "positive_rows": self.positive_rows,
            "metrics": {name: summary.to_dict() for name, summary in self.metrics.items()},
        }


def build_baseline_estimators(random_seed: int = 42) -> dict[str, ClassifierMixin]:
    """Return intentionally simple untuned baseline estimators."""

    return {
        "dummy": DummyClassifier(strategy="prior", random_state=random_seed),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def build_model_pipeline(
    X_train: pd.DataFrame,
    estimator: ClassifierMixin,
) -> Pipeline:
    """Combine leakage-safe preprocessing and an estimator."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ]
    )


def build_scoring() -> dict[str, object]:
    """Return scoring rules used consistently across baseline models."""

    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
    }


def evaluate_baseline(
    name: str,
    estimator: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> BaselineResult:
    """Evaluate one model on training data using stratified CV."""

    if folds < 2:
        raise ValueError("folds must be at least 2")

    pipeline = build_model_pipeline(X_train, estimator)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        scoring=build_scoring(),
        cv=cv,
        return_train_score=False,
        n_jobs=None,
    )

    metrics: dict[str, MetricSummary] = {}
    for metric in METRIC_NAMES:
        values = np.asarray(scores[f"test_{metric}"], dtype=float)
        metrics[metric] = MetricSummary(
            mean=float(values.mean()),
            std=float(values.std(ddof=0)),
            folds=tuple(float(value) for value in values),
        )

    return BaselineResult(
        model=name,
        folds=folds,
        training_rows=len(X_train),
        positive_rows=int(y_train.sum()),
        metrics=metrics,
    )


def evaluate_all_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> list[BaselineResult]:
    """Evaluate all Stage 4.1 models without touching the final holdout."""

    estimators = build_baseline_estimators(random_seed)
    return [
        evaluate_baseline(
            name,
            estimator,
            X_train,
            y_train,
            folds=folds,
            random_seed=random_seed,
        )
        for name, estimator in estimators.items()
    ]


def main() -> None:
    """Run Stage 4.1 baselines on the training partition and print JSON."""

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
    payload = {
        "evaluation": "training-only stratified cross-validation",
        "final_holdout_rows": len(split.X_test),
        "holdout_evaluated": False,
        "models": [result.to_dict() for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
