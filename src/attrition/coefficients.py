"""Global coefficient analysis for the frozen tuned Logistic Regression."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from attrition.calibration import build_tuned_logistic
from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.split import create_stratified_split

DEFAULT_TOP_N = 15
DEFAULT_OUTPUT_DIR = Path("reports/generated")


def _clean_feature_name(name: str) -> str:
    """Convert ColumnTransformer output names into readable labels."""

    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def fit_explainable_base_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_seed: int = 42,
) -> Pipeline:
    """Fit the frozen Stage 6 base Logistic Regression on training data only."""

    estimator = build_tuned_logistic(X_train, random_seed=random_seed)
    estimator.fit(X_train, y_train)
    return estimator


def coefficient_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Map fitted Logistic Regression coefficients to transformed feature names."""

    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("pipeline model must be fitted before coefficient extraction")

    feature_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = np.asarray(model.coef_, dtype=float).ravel()

    if len(feature_names) != len(coefficients):
        raise ValueError("feature-name and coefficient counts do not match")

    table = pd.DataFrame(
        {
            "feature": [_clean_feature_name(str(name)) for name in feature_names],
            "transformed_feature": [str(name) for name in feature_names],
            "coefficient": coefficients,
        }
    )
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["absolute_coefficient"] = table["coefficient"].abs()
    table["direction"] = np.where(
        table["coefficient"] > 0,
        "higher_attrition_log_odds",
        np.where(
            table["coefficient"] < 0,
            "lower_attrition_log_odds",
            "neutral",
        ),
    )
    return table.sort_values(
        ["absolute_coefficient", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )


def ranked_drivers(
    table: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, list[dict[str, object]]]:
    """Return strongest positive and negative coefficient directions."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    columns = [
        "feature",
        "coefficient",
        "odds_ratio",
        "absolute_coefficient",
        "direction",
    ]
    positive = (
        table.loc[table["coefficient"] > 0]
        .nlargest(top_n, "coefficient")[columns]
        .to_dict(orient="records")
    )
    negative = (
        table.loc[table["coefficient"] < 0]
        .nsmallest(top_n, "coefficient")[columns]
        .to_dict(orient="records")
    )
    return {
        "higher_attrition_log_odds": positive,
        "lower_attrition_log_odds": negative,
    }


def build_coefficient_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_seed: int = 42,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, object]:
    """Fit the frozen base model and construct global coefficient diagnostics."""

    fitted = fit_explainable_base_model(
        X_train,
        y_train,
        random_seed=random_seed,
    )
    table = coefficient_table(fitted)
    drivers = ranked_drivers(table, top_n=top_n)
    model = fitted.named_steps["model"]

    return {
        "model": {
            "name": "logistic_regression",
            "C": float(model.C),
            "penalty": str(model.penalty),
            "solver": str(model.solver),
            "intercept": float(np.asarray(model.intercept_).ravel()[0]),
        },
        "training_rows": len(X_train),
        "transformed_feature_count": len(table),
        "explanation_scope": (
            "Coefficients explain the frozen base logistic decision function fitted on "
            "training data. Isotonic calibration is a probability post-processing layer "
            "and does not have one global coefficient vector."
        ),
        "interpretation": (
            "Positive coefficients increase attrition log-odds; negative coefficients "
            "decrease attrition log-odds. Magnitudes are not causal effects."
        ),
        "drivers": drivers,
        "table": table,
    }


def save_coefficient_analysis(
    analysis: dict[str, object],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Save the full coefficient table and concise ranked-driver summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "logistic_coefficients.csv"
    json_path = output_dir / "coefficient_analysis.json"

    table = analysis["table"]
    table.to_csv(csv_path, index=False)

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
        "coefficient_csv": str(csv_path),
        "summary_json": str(json_path),
    }


def main() -> None:
    """Run Stage 7.1 coefficient analysis on the training partition only."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )
    analysis = build_coefficient_analysis(
        split.X_train,
        split.y_train,
        random_seed=settings.random_seed,
    )
    artifacts = save_coefficient_analysis(analysis)

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
