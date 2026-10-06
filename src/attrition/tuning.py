"""Two-pass hyperparameter tuning on training data only."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.imbalance import build_imbalance_experiments
from attrition.modeling import build_model_pipeline, build_scoring
from attrition.split import create_stratified_split

TUNING_CANDIDATES = (
    "logistic_baseline",
    "logistic_class_weight",
    "random_forest_smote",
)
PRIMARY_METRIC = "pr_auc"
LOGISTIC_C_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


@dataclass(frozen=True)
class SearchSummary:
    """Serializable summary for one search pass."""

    best_score: float
    best_params: dict[str, object]
    candidates_evaluated: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TuningResult:
    """Randomized-search and targeted-grid results for one candidate."""

    candidate: str
    randomized: SearchSummary
    grid: SearchSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "randomized": self.randomized.to_dict(),
            "grid": self.grid.to_dict(),
        }


def build_tuning_candidates(
    X_train: pd.DataFrame,
    *,
    random_seed: int = 42,
) -> dict[str, object]:
    """Build the three Stage 6.1 candidates without fitting them."""

    logistic = build_model_pipeline(
        X_train,
        LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            random_state=random_seed,
        ),
    )
    weighted = build_model_pipeline(
        X_train,
        LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_seed,
        ),
    )
    rf_smote = clone(
        build_imbalance_experiments(
            X_train,
            random_seed=random_seed,
        )["random_forest_smote"]
    )
    rf_smote.set_params(model__n_jobs=1)

    return {
        "logistic_baseline": logistic,
        "logistic_class_weight": weighted,
        "random_forest_smote": rf_smote,
    }


def randomized_search_spaces() -> dict[str, dict[str, list[object]]]:
    """Return broad but bounded search spaces for the randomized pass."""

    logistic_space = {
        "model__C": list(LOGISTIC_C_VALUES),
        "model__penalty": ["l1", "l2"],
    }
    return {
        "logistic_baseline": logistic_space,
        "logistic_class_weight": logistic_space,
        "random_forest_smote": {
            "sampler__k_neighbors": [3, 5, 7],
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
    }


def targeted_grid(
    candidate: str,
    best_params: dict[str, object],
) -> dict[str, list[object]]:
    """Build a smaller grid around the randomized-search winner."""

    if candidate in {"logistic_baseline", "logistic_class_weight"}:
        best_c = float(best_params["model__C"])
        index = LOGISTIC_C_VALUES.index(best_c)
        lower = max(0, index - 1)
        upper = min(len(LOGISTIC_C_VALUES), index + 2)
        return {
            "model__C": list(LOGISTIC_C_VALUES[lower:upper]),
            "model__penalty": [best_params["model__penalty"]],
        }

    if candidate == "random_forest_smote":
        depth = best_params["model__max_depth"]
        depth_values = [depth]
        if isinstance(depth, int):
            depth_values = sorted({max(2, depth - 5), depth, depth + 5})

        estimators = int(best_params["model__n_estimators"])
        estimator_values = sorted({max(100, estimators - 100), estimators, estimators + 100})

        return {
            "sampler__k_neighbors": [best_params["sampler__k_neighbors"]],
            "model__n_estimators": estimator_values,
            "model__max_depth": depth_values,
            "model__min_samples_split": [best_params["model__min_samples_split"]],
            "model__min_samples_leaf": [best_params["model__min_samples_leaf"]],
            "model__max_features": [best_params["model__max_features"]],
        }

    raise ValueError(f"unknown tuning candidate: {candidate}")


def _cv(folds: int, random_seed: int) -> StratifiedKFold:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)


def run_randomized_search(
    candidate: str,
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
    n_iter: int | None = None,
) -> RandomizedSearchCV:
    """Run the broad training-only tuning pass."""

    spaces = randomized_search_spaces()
    if candidate not in spaces:
        raise ValueError(f"unknown tuning candidate: {candidate}")

    iterations = n_iter
    if iterations is None:
        iterations = 12 if candidate.startswith("logistic") else 16

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=spaces[candidate],
        n_iter=iterations,
        scoring=build_scoring(),
        refit=PRIMARY_METRIC,
        cv=_cv(folds, random_seed),
        random_state=random_seed,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def run_targeted_grid_search(
    candidate: str,
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, object],
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> GridSearchCV:
    """Refine the randomized-search winner with a narrow grid."""

    search = GridSearchCV(
        estimator=estimator,
        param_grid=targeted_grid(candidate, best_params),
        scoring=build_scoring(),
        refit=PRIMARY_METRIC,
        cv=_cv(folds, random_seed),
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def _summary(search: RandomizedSearchCV | GridSearchCV) -> SearchSummary:
    return SearchSummary(
        best_score=float(search.best_score_),
        best_params=dict(search.best_params_),
        candidates_evaluated=len(search.cv_results_["params"]),
    )


def tune_candidate(
    candidate: str,
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> TuningResult:
    """Run randomized search followed by targeted grid refinement."""

    randomized = run_randomized_search(
        candidate,
        estimator,
        X_train,
        y_train,
        folds=folds,
        random_seed=random_seed,
    )
    grid = run_targeted_grid_search(
        candidate,
        estimator,
        X_train,
        y_train,
        randomized.best_params_,
        folds=folds,
        random_seed=random_seed,
    )
    return TuningResult(
        candidate=candidate,
        randomized=_summary(randomized),
        grid=_summary(grid),
    )


def tune_all_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> list[TuningResult]:
    """Tune all Stage 6.1 candidates using training data only."""

    candidates = build_tuning_candidates(X_train, random_seed=random_seed)
    return [
        tune_candidate(
            name,
            estimator,
            X_train,
            y_train,
            folds=folds,
            random_seed=random_seed,
        )
        for name, estimator in candidates.items()
    ]


def main() -> None:
    """Run Stage 6.1 without evaluating the final holdout."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )
    results = tune_all_candidates(
        split.X_train,
        split.y_train,
        folds=5,
        random_seed=settings.random_seed,
    )
    payload = {
        "evaluation": "training-only two-pass hyperparameter tuning",
        "primary_metric": PRIMARY_METRIC,
        "final_holdout_rows": len(split.X_test),
        "holdout_evaluated": False,
        "results": [result.to_dict() for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
