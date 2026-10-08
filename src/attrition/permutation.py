"""Cross-validated permutation importance for the frozen final policy."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.final_evaluation import build_final_estimator
from attrition.split import create_stratified_split

DEFAULT_FOLDS = 5
DEFAULT_REPEATS = 8
DEFAULT_TOP_N = 15
PRIMARY_IMPORTANCE_METRIC = "average_precision"
SECONDARY_IMPORTANCE_METRIC = "roc_auc"
DEFAULT_OUTPUT_DIR = Path("reports/generated")


@dataclass(frozen=True)
class ScoreSummary:
    """Mean and variation of validation scores across outer folds."""

    mean: float
    std: float
    folds: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _importance_table(
    feature_names: list[str],
    metric_importances: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    """Aggregate feature importance values across folds and repeats."""

    rows: dict[str, object] = {"feature": feature_names}
    for metric, arrays in metric_importances.items():
        combined = np.concatenate(arrays, axis=1)
        rows[f"{metric}_importance_mean"] = combined.mean(axis=1)
        rows[f"{metric}_importance_std"] = combined.std(axis=1, ddof=0)

    table = pd.DataFrame(rows)
    return table.sort_values(
        f"{PRIMARY_IMPORTANCE_METRIC}_importance_mean",
        ascending=False,
        ignore_index=True,
    )


def cross_validated_permutation_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = DEFAULT_FOLDS,
    repeats: int = DEFAULT_REPEATS,
    random_seed: int = 42,
) -> dict[str, object]:
    """Measure permutation importance on held-out folds of training data."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_seed,
    )
    metric_importances: dict[str, list[np.ndarray]] = {
        PRIMARY_IMPORTANCE_METRIC: [],
        SECONDARY_IMPORTANCE_METRIC: [],
    }
    fold_average_precision: list[float] = []
    fold_roc_auc: list[float] = []

    for fold_index, (fit_indices, validation_indices) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        X_fit = X_train.iloc[fit_indices]
        y_fit = y_train.iloc[fit_indices]
        X_validation = X_train.iloc[validation_indices]
        y_validation = y_train.iloc[validation_indices]

        estimator = build_final_estimator(
            X_fit,
            random_seed=random_seed,
        )
        estimator.fit(X_fit, y_fit)

        probabilities = estimator.predict_proba(X_validation)[:, 1]
        fold_average_precision.append(
            float(average_precision_score(y_validation, probabilities))
        )
        fold_roc_auc.append(float(roc_auc_score(y_validation, probabilities)))

        result = permutation_importance(
            estimator,
            X_validation,
            y_validation,
            scoring={
                PRIMARY_IMPORTANCE_METRIC: "average_precision",
                SECONDARY_IMPORTANCE_METRIC: "roc_auc",
            },
            n_repeats=repeats,
            random_state=random_seed + fold_index,
            n_jobs=-1,
        )
        for metric in metric_importances:
            metric_importances[metric].append(
                np.asarray(result[metric].importances, dtype=float)
            )

    table = _importance_table(list(X_train.columns), metric_importances)
    ap_values = np.asarray(fold_average_precision, dtype=float)
    roc_values = np.asarray(fold_roc_auc, dtype=float)

    return {
        "table": table,
        "folds": folds,
        "repeats_per_fold": repeats,
        "scoring": {
            "primary": PRIMARY_IMPORTANCE_METRIC,
            "secondary": SECONDARY_IMPORTANCE_METRIC,
        },
        "validation_scores": {
            "average_precision": ScoreSummary(
                mean=float(ap_values.mean()),
                std=float(ap_values.std(ddof=0)),
                folds=tuple(float(value) for value in ap_values),
            ).to_dict(),
            "roc_auc": ScoreSummary(
                mean=float(roc_values.mean()),
                std=float(roc_values.std(ddof=0)),
                folds=tuple(float(value) for value in roc_values),
            ).to_dict(),
        },
    }


def top_features(
    table: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, object]]:
    """Return the strongest features by mean Average Precision decrease."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    columns = [
        "feature",
        "average_precision_importance_mean",
        "average_precision_importance_std",
        "roc_auc_importance_mean",
        "roc_auc_importance_std",
    ]
    return table.head(top_n)[columns].to_dict(orient="records")


def build_permutation_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = DEFAULT_FOLDS,
    repeats: int = DEFAULT_REPEATS,
    random_seed: int = 42,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, object]:
    """Run Stage 7.2 and package the global importance report."""

    result = cross_validated_permutation_importance(
        X_train,
        y_train,
        folds=folds,
        repeats=repeats,
        random_seed=random_seed,
    )
    table = result["table"]

    return {
        "method": "cross-validated permutation importance",
        "model_scope": (
            "Frozen calibrated final policy evaluated on held-out folds of the "
            "training partition only."
        ),
        "importance_definition": (
            "Mean decrease in validation score after shuffling one original input "
            "feature while leaving the fitted estimator unchanged."
        ),
        "limitations": (
            "Importance is predictive rather than causal and can be diluted across "
            "correlated or redundant features."
        ),
        "training_rows": len(X_train),
        "original_feature_count": X_train.shape[1],
        "folds": result["folds"],
        "repeats_per_fold": result["repeats_per_fold"],
        "scoring": result["scoring"],
        "validation_scores": result["validation_scores"],
        "top_features": top_features(table, top_n=top_n),
        "table": table,
    }


def save_permutation_analysis(
    analysis: dict[str, object],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Persist the full importance table and concise JSON summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "permutation_importance.csv"
    json_path = output_dir / "permutation_importance.json"

    analysis["table"].to_csv(csv_path, index=False)
    serializable = {
        key: value
        for key, value in analysis.items()
        if key != "table"
    }
    json_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "importance_csv": str(csv_path),
        "summary_json": str(json_path),
    }


def main() -> None:
    """Run Stage 7.2 without reusing the consumed final holdout."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )
    analysis = build_permutation_analysis(
        split.X_train,
        split.y_train,
        random_seed=settings.random_seed,
    )
    artifacts = save_permutation_analysis(analysis)

    payload = {
        key: value
        for key, value in analysis.items()
        if key != "table"
    }
    payload["artifacts"] = artifacts
    payload["holdout_used_for_explainability"] = False
    payload["holdout_rows_excluded"] = len(split.X_test)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
