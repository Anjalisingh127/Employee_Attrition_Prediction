"""Local SHAP explanations for individual training-partition employees."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from attrition.coefficients import fit_explainable_base_model
from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.final_evaluation import FINAL_THRESHOLD, build_final_estimator
from attrition.shap_global import build_shap_explanation
from attrition.split import create_stratified_split

DEFAULT_ROW_POSITION = 0
DEFAULT_TOP_N = 10
DEFAULT_OUTPUT_DIR = Path("reports/generated")


def _python_scalar(value: object) -> object:
    """Convert NumPy/Pandas scalar values into JSON-friendly Python values."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def local_contribution_table(
    explanation: shap.Explanation,
    row_position: int,
) -> pd.DataFrame:
    """Return ranked SHAP contributions for one explained row."""

    if row_position < 0 or row_position >= len(explanation):
        raise IndexError("row_position is outside the explained dataset")

    row = explanation[row_position]
    values = np.asarray(row.values, dtype=float).ravel()
    data = np.asarray(row.data, dtype=float).ravel()
    names = list(row.feature_names)

    table = pd.DataFrame(
        {
            "feature": names,
            "transformed_value": data,
            "shap_value": values,
        }
    )
    table["absolute_shap"] = table["shap_value"].abs()
    table["direction"] = np.where(
        table["shap_value"] > 0,
        "increases_attrition_score",
        np.where(
            table["shap_value"] < 0,
            "decreases_attrition_score",
            "neutral",
        ),
    )
    return table.sort_values(
        ["absolute_shap", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )


def top_local_contributions(
    table: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, list[dict[str, object]]]:
    """Return strongest local risk-increasing and risk-reducing contributions."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    columns = [
        "feature",
        "transformed_value",
        "shap_value",
        "absolute_shap",
        "direction",
    ]
    increasing = (
        table.loc[table["shap_value"] > 0]
        .nlargest(top_n, "shap_value")[columns]
        .to_dict(orient="records")
    )
    reducing = (
        table.loc[table["shap_value"] < 0]
        .nsmallest(top_n, "shap_value")[columns]
        .to_dict(orient="records")
    )
    return {
        "risk_increasing": increasing,
        "risk_reducing": reducing,
    }


def build_local_shap_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    row_position: int = DEFAULT_ROW_POSITION,
    random_seed: int = 42,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, object]:
    """Explain one training-partition employee under the frozen model policy."""

    if row_position < 0 or row_position >= len(X_train):
        raise IndexError("row_position is outside the training partition")

    base_model = fit_explainable_base_model(
        X_train,
        y_train,
        random_seed=random_seed,
    )
    explanation = build_shap_explanation(base_model, X_train)
    table = local_contribution_table(explanation, row_position)

    selected_X = X_train.iloc[[row_position]]
    transformed = base_model.named_steps["preprocessor"].transform(selected_X)
    raw_score = float(base_model.named_steps["model"].decision_function(transformed)[0])
    base_probability = float(base_model.predict_proba(selected_X)[0, 1])

    calibrated_model = build_final_estimator(
        X_train,
        random_seed=random_seed,
    )
    calibrated_model.fit(X_train, y_train)
    calibrated_probability = float(calibrated_model.predict_proba(selected_X)[0, 1])
    prediction = int(calibrated_probability >= FINAL_THRESHOLD)

    row_explanation = explanation[row_position]
    base_value = float(np.asarray(row_explanation.base_values).reshape(-1)[0])
    shap_sum = float(np.asarray(row_explanation.values, dtype=float).sum())
    reconstructed_score = base_value + shap_sum

    raw_record = {
        str(key): _python_scalar(value)
        for key, value in X_train.iloc[row_position].to_dict().items()
    }

    return {
        "row_position": row_position,
        "source_row_index": _python_scalar(X_train.index[row_position]),
        "actual_training_label": int(y_train.iloc[row_position]),
        "model": {
            "name": "logistic_regression",
            "C": 0.3,
            "penalty": "l2",
            "solver": "liblinear",
        },
        "prediction": {
            "base_logistic_probability": base_probability,
            "calibrated_probability": calibrated_probability,
            "threshold": FINAL_THRESHOLD,
            "predicted_attrition": prediction,
            "scope": (
                "Training-partition demonstration only; this probability is not an "
                "additional evaluation metric."
            ),
        },
        "shap_additivity": {
            "base_value": base_value,
            "shap_sum": shap_sum,
            "reconstructed_logit": reconstructed_score,
            "model_logit": raw_score,
            "absolute_error": abs(reconstructed_score - raw_score),
        },
        "explanation_scope": (
            "SHAP decomposes the frozen underlying Logistic Regression score. "
            "It does not decompose the isotonic-calibrated probability."
        ),
        "interpretation": (
            "Positive SHAP values push this employee's attrition score upward; "
            "negative values push it downward. Contributions are model explanations, "
            "not causal HR effects."
        ),
        "top_contributions": top_local_contributions(table, top_n=top_n),
        "raw_input": raw_record,
        "table": table,
        "row_explanation": row_explanation,
    }


def save_local_shap_analysis(
    analysis: dict[str, object],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Persist one employee's local explanation and waterfall plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"row_{analysis['row_position']}"
    csv_path = output_dir / f"shap_local_{suffix}.csv"
    json_path = output_dir / f"shap_local_{suffix}.json"
    waterfall_path = output_dir / f"shap_local_{suffix}_waterfall.png"

    analysis["table"].to_csv(csv_path, index=False)
    serializable = {
        key: value
        for key, value in analysis.items()
        if key not in {"table", "row_explanation"}
    }
    json_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    shap.plots.waterfall(
        analysis["row_explanation"],
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()

    return {
        "contribution_csv": str(csv_path),
        "summary_json": str(json_path),
        "waterfall_plot": str(waterfall_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain one training-partition employee with local SHAP values."
    )
    parser.add_argument(
        "--row-position",
        type=int,
        default=DEFAULT_ROW_POSITION,
        help="Zero-based position within the training partition (default: 0).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of increasing and reducing contributions to report (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    """Run Stage 7.4 local SHAP explainability without using the final holdout."""

    args = _parse_args()
    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )

    analysis = build_local_shap_analysis(
        split.X_train,
        split.y_train,
        row_position=args.row_position,
        random_seed=settings.random_seed,
        top_n=args.top_n,
    )
    artifacts = save_local_shap_analysis(analysis)

    payload = {
        key: value
        for key, value in analysis.items()
        if key not in {"table", "row_explanation", "raw_input"}
    }
    payload["artifacts"] = artifacts
    payload["holdout_used_for_explainability"] = False
    payload["holdout_rows_excluded"] = len(split.X_test)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
