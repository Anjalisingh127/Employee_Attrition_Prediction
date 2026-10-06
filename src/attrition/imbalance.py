"""Leakage-safe class-imbalance experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.modeling import BaselineResult, METRIC_NAMES, MetricSummary, build_scoring
from attrition.preprocessing import build_preprocessor
from attrition.split import create_stratified_split

EXPERIMENT_NAMES = (
    "logistic_class_weight",
    "logistic_random_oversampling",
    "logistic_smote",
    "random_forest_class_weight",
    "random_forest_random_oversampling",
    "random_forest_smote",
)


def build_imbalance_experiments(
    X_train: pd.DataFrame,
    *,
    random_seed: int = 42,
) -> dict[str, ImbPipeline]:
    """Build unfitted pipelines for class weighting and fold-local resampling."""

    preprocessor = build_preprocessor(X_train)

    def pipeline(
        estimator: ClassifierMixin,
        sampler: object | None = None,
    ) -> ImbPipeline:
        steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.append(("model", estimator))
        return ImbPipeline(steps=steps)

    return {
        "logistic_class_weight": pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_seed,
            )
        ),
        "logistic_random_oversampling": pipeline(
            LogisticRegression(max_iter=2000, random_state=random_seed),
            RandomOverSampler(random_state=random_seed),
        ),
        "logistic_smote": pipeline(
            LogisticRegression(max_iter=2000, random_state=random_seed),
            SMOTE(random_state=random_seed),
        ),
        "random_forest_class_weight": pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
            )
        ),
        "random_forest_random_oversampling": pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_seed,
                n_jobs=-1,
            ),
            RandomOverSampler(random_state=random_seed),
        ),
        "random_forest_smote": pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_seed,
                n_jobs=-1,
            ),
            SMOTE(random_state=random_seed),
        ),
    }


def evaluate_imbalance_experiment(
    name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> BaselineResult:
    """Evaluate one imbalance strategy using training-only stratified CV."""

    if folds < 2:
        raise ValueError("folds must be at least 2")

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


def evaluate_all_imbalance_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> list[BaselineResult]:
    """Evaluate every Stage 5 imbalance strategy."""

    experiments = build_imbalance_experiments(X_train, random_seed=random_seed)
    return [
        evaluate_imbalance_experiment(
            name,
            experiment,
            X_train,
            y_train,
            folds=folds,
            random_seed=random_seed,
        )
        for name, experiment in experiments.items()
    ]


def main() -> None:
    """Run Stage 5 experiments without evaluating the final holdout."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )
    results = evaluate_all_imbalance_experiments(
        split.X_train,
        split.y_train,
        folds=5,
        random_seed=settings.random_seed,
    )
    payload = {
        "evaluation": "training-only imbalance experiments with stratified CV",
        "final_holdout_rows": len(split.X_test),
        "holdout_evaluated": False,
        "experiments": [result.to_dict() for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
