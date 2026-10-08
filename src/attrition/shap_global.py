"""Global SHAP analysis for the frozen tuned Logistic Regression."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from attrition.coefficients import _clean_feature_name, fit_explainable_base_model
from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.split import create_stratified_split

DEFAULT_TOP_N = 15
DEFAULT_OUTPUT_DIR = Path("reports/generated")


def transformed_training_data(
    fitted_pipeline,
    X_train: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Transform training inputs and return aligned readable feature names."""

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.transform(X_train), dtype=float)
    names = [
        _clean_feature_name(str(name))
        for name in preprocessor.get_feature_names_out()
    ]

    if transformed.shape[1] != len(names):
        raise ValueError("transformed feature count does not match feature names")
    return transformed, names


def build_shap_explanation(
    fitted_pipeline,
    X_train: pd.DataFrame,
) -> shap.Explanation:
    """Explain the frozen base Logistic Regression on training data only."""

    transformed, feature_names = transformed_training_data(
        fitted_pipeline,
        X_train,
    )
    model = fitted_pipeline.named_steps["model"]
    masker = shap.maskers.Independent(
        transformed,
        max_samples=len(transformed),
    )
    explainer = shap.LinearExplainer(model, masker)
    explanation = explainer(transformed)

    if explanation.values.shape != transformed.shape:
        raise ValueError("SHAP value matrix does not match transformed feature matrix")

    return shap.Explanation(
        values=np.asarray(explanation.values, dtype=float),
        base_values=np.asarray(explanation.base_values, dtype=float),
        data=transformed,
        feature_names=feature_names,
    )


def global_importance_table(explanation: shap.Explanation) -> pd.DataFrame:
    """Aggregate global SHAP magnitude and signed direction by transformed feature."""

    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 2:
        raise ValueError("global SHAP analysis requires a 2D explanation matrix")

    table = pd.DataFrame(
        {
            "feature": list(explanation.feature_names),
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_signed_shap": values.mean(axis=0),
            "positive_share": (values > 0).mean(axis=0),
            "negative_share": (values < 0).mean(axis=0),
        }
    )
    return table.sort_values(
        ["mean_abs_shap", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )


def top_global_features(
    table: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, object]]:
    """Return the top transformed features by mean absolute SHAP value."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    columns = [
        "feature",
        "mean_abs_shap",
        "mean_signed_shap",
        "positive_share",
        "negative_share",
    ]
    return table.head(top_n)[columns].to_dict(orient="records")


def build_global_shap_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_seed: int = 42,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, object]:
    """Fit the frozen base model and construct global SHAP diagnostics."""

    fitted = fit_explainable_base_model(
        X_train,
        y_train,
        random_seed=random_seed,
    )
    explanation = build_shap_explanation(fitted, X_train)
    table = global_importance_table(explanation)

    return {
        "method": "SHAP LinearExplainer",
        "training_rows": len(X_train),
        "transformed_feature_count": len(table),
        "model_scope": (
            "Frozen tuned Logistic Regression decision function fitted on the "
            "training partition only."
        ),
        "calibration_scope": (
            "Isotonic calibration is a separate probability post-processing layer "
            "and is not represented by this linear SHAP explanation."
        ),
        "interpretation": (
            "mean_abs_shap measures average global contribution magnitude in model "
            "output space; signed SHAP values show contribution direction for each row."
        ),
        "limitations": (
            "SHAP values explain model behavior rather than causal HR effects. "
            "Correlated and one-hot encoded features can share or redistribute attribution."
        ),
        "top_features": top_global_features(table, top_n=top_n),
        "table": table,
        "explanation": explanation,
    }


def save_global_shap_analysis(
    analysis: dict[str, object],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Persist SHAP tables, summary metadata, and global plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "shap_global_importance.csv"
    json_path = output_dir / "shap_global_summary.json"
    bar_path = output_dir / "shap_global_bar.png"
    beeswarm_path = output_dir / "shap_global_beeswarm.png"

    analysis["table"].to_csv(csv_path, index=False)

    serializable = {
        key: value
        for key, value in analysis.items()
        if key not in {"table", "explanation"}
    }
    json_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    explanation = analysis["explanation"]

    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()

    return {
        "importance_csv": str(csv_path),
        "summary_json": str(json_path),
        "bar_plot": str(bar_path),
        "beeswarm_plot": str(beeswarm_path),
    }


def main() -> None:
    """Run Stage 7.3 without reusing the consumed final holdout."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )
    analysis = build_global_shap_analysis(
        split.X_train,
        split.y_train,
        random_seed=settings.random_seed,
    )
    artifacts = save_global_shap_analysis(analysis)

    payload = {
        key: value
        for key, value in analysis.items()
        if key not in {"table", "explanation"}
    }
    payload["artifacts"] = artifacts
    payload["holdout_used_for_explainability"] = False
    payload["holdout_rows_excluded"] = len(split.X_test)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
