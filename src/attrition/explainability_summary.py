"""Recruiter-facing synthesis of the Stage 7 explainability evidence."""

import json
from pathlib import Path

import pandas as pd

from attrition.coefficients import build_coefficient_analysis
from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.permutation import build_permutation_analysis
from attrition.shap_global import build_global_shap_analysis
from attrition.split import create_stratified_split

DEFAULT_TOP_N = 15
DEFAULT_OUTPUT_DIR = Path("reports/generated")


def feature_family(feature: str, original_features: list[str]) -> str:
    """Map transformed terms back to their original business feature family."""

    matches = [
        candidate
        for candidate in original_features
        if feature == candidate or feature.startswith(f"{candidate}_")
    ]
    if matches:
        return max(matches, key=len)
    return feature


def aggregate_transformed_importance(
    table: pd.DataFrame,
    *,
    value_column: str,
    original_features: list[str],
) -> pd.DataFrame:
    """Aggregate transformed feature importance into original feature families."""

    grouped = table[["feature", value_column]].copy()
    grouped["feature_family"] = grouped["feature"].map(
        lambda value: feature_family(str(value), original_features)
    )
    result = (
        grouped.groupby("feature_family", as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False, ignore_index=True)
    )
    return result


def rank_table(
    table: pd.DataFrame,
    *,
    value_column: str,
    method: str,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Return a transparent rank list for one explainability method."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    ranked = table.sort_values(value_column, ascending=False).head(top_n).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["method"] = method
    return ranked.rename(columns={value_column: "importance"})[
        ["feature_family", "method", "rank", "importance"]
    ]


def consensus_table(
    coefficient_table: pd.DataFrame,
    permutation_table: pd.DataFrame,
    shap_table: pd.DataFrame,
    original_features: list[str],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Build cross-method consensus without inventing a weighted score."""

    coefficient_family = aggregate_transformed_importance(
        coefficient_table,
        value_column="absolute_coefficient",
        original_features=original_features,
    )
    shap_family = aggregate_transformed_importance(
        shap_table,
        value_column="mean_abs_shap",
        original_features=original_features,
    )
    permutation_family = permutation_table.rename(
        columns={
            "feature": "feature_family",
            "average_precision_importance_mean": "importance",
        }
    )[["feature_family", "importance"]]

    ranked = pd.concat(
        [
            rank_table(
                coefficient_family,
                value_column="absolute_coefficient",
                method="coefficient",
                top_n=top_n,
            ),
            rank_table(
                permutation_family,
                value_column="importance",
                method="permutation",
                top_n=top_n,
            ),
            rank_table(
                shap_family,
                value_column="mean_abs_shap",
                method="shap",
                top_n=top_n,
            ),
        ],
        ignore_index=True,
    )

    consensus = (
        ranked.groupby("feature_family", as_index=False)
        .agg(
            methods=("method", "nunique"),
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
            method_list=("method", lambda values: ", ".join(sorted(set(values)))),
        )
        .sort_values(
            ["methods", "mean_rank", "best_rank", "feature_family"],
            ascending=[False, True, True, True],
            ignore_index=True,
        )
    )
    return consensus


def strongest_directional_terms(
    coefficient_table: pd.DataFrame,
    *,
    top_n: int = 5,
) -> dict[str, list[dict[str, object]]]:
    """Return a small set of defensible signed coefficient terms."""

    columns = ["feature", "coefficient", "odds_ratio"]
    increasing = (
        coefficient_table.loc[coefficient_table["coefficient"] > 0]
        .nlargest(top_n, "coefficient")[columns]
        .to_dict(orient="records")
    )
    reducing = (
        coefficient_table.loc[coefficient_table["coefficient"] < 0]
        .nsmallest(top_n, "coefficient")[columns]
        .to_dict(orient="records")
    )
    return {"higher_attrition_log_odds": increasing, "lower_attrition_log_odds": reducing}


def build_explainability_summary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_seed: int = 42,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, object]:
    """Regenerate Stage 7 evidence and synthesize cross-method findings."""

    coefficient = build_coefficient_analysis(
        X_train,
        y_train,
        random_seed=random_seed,
        top_n=top_n,
    )
    permutation = build_permutation_analysis(
        X_train,
        y_train,
        random_seed=random_seed,
        top_n=top_n,
    )
    shap_global = build_global_shap_analysis(
        X_train,
        y_train,
        random_seed=random_seed,
        top_n=top_n,
    )

    consensus = consensus_table(
        coefficient["table"],
        permutation["table"],
        shap_global["table"],
        list(X_train.columns),
        top_n=top_n,
    )
    cross_method = consensus.loc[consensus["methods"] >= 2].copy()

    return {
        "training_rows": len(X_train),
        "methods": [
            "logistic_regression_coefficients",
            "cross_validated_permutation_importance",
            "global_shap",
        ],
        "consensus_rule": (
            f"Feature family appears in the top {top_n} of at least two independent "
            "global explanation methods. No weighted composite score is used."
        ),
        "cross_method_features": cross_method.to_dict(orient="records"),
        "directional_terms": strongest_directional_terms(coefficient["table"]),
        "permutation_validation": permutation["validation_scores"],
        "limitations": [
            "All explainability findings describe model behavior, not causal HR effects.",
            "Correlated or redundant variables can share importance.",
            "Coefficient and SHAP explanations describe the underlying Logistic Regression; "
            "isotonic calibration is a separate probability mapping.",
            "The consumed final holdout is not reused for Stage 7 explainability.",
        ],
        "consensus_table": consensus,
    }


def render_recruiter_report(summary: dict[str, object]) -> str:
    """Render a concise Markdown report suitable for repository review."""

    features = summary["cross_method_features"][:10]
    feature_lines = "\n".join(
        f"- **{item['feature_family']}** — supported by {item['methods']} methods "
        f"({item['method_list']}); mean top-tier rank {item['mean_rank']:.1f}."
        for item in features
    )

    higher = summary["directional_terms"]["higher_attrition_log_odds"]
    lower = summary["directional_terms"]["lower_attrition_log_odds"]
    higher_lines = "\n".join(
        f"- **{item['feature']}**: coefficient {item['coefficient']:.3f}, "
        f"odds ratio {item['odds_ratio']:.2f}."
        for item in higher
    )
    lower_lines = "\n".join(
        f"- **{item['feature']}**: coefficient {item['coefficient']:.3f}, "
        f"odds ratio {item['odds_ratio']:.2f}."
        for item in lower
    )

    ap = summary["permutation_validation"]["average_precision"]
    roc = summary["permutation_validation"]["roc_auc"]

    return f"""# Explainability Summary

## Executive summary

The frozen employee-attrition model is explained through three complementary global methods: Logistic Regression coefficients, cross-validated permutation importance, and SHAP. Local SHAP explanations are also available for individual training-partition demonstrations.

The strongest conclusions are based on **cross-method agreement**, not a single importance chart. The report intentionally avoids causal language: these features explain model behavior in the IBM benchmark dataset and should not be interpreted as proven causes of employee attrition.

## Cross-method signals

{feature_lines}

## Directional model terms

### Higher modeled attrition log-odds

{higher_lines}

### Lower modeled attrition log-odds

{lower_lines}

These signed terms come from the frozen Logistic Regression decision function. Categorical one-hot terms should be interpreted as encoded category contributions rather than standalone interventions.

## Validation context

Permutation importance was measured on held-out folds of the training partition. The frozen calibrated policy achieved mean Average Precision **{ap['mean']:.3f} ± {ap['std']:.3f}** and mean ROC-AUC **{roc['mean']:.3f} ± {roc['std']:.3f}** across those folds.

## Explainability architecture

- **Coefficients:** signed linear weights and odds-ratio interpretation.
- **Permutation importance:** predictive dependence on original business features through the full calibrated pipeline.
- **Global SHAP:** average transformed-feature contribution magnitude plus row-level direction.
- **Local SHAP:** employee-level risk-increasing and risk-reducing contributions with an additivity check.

## Guardrails

- Explainability is descriptive of model behavior, not causal HR evidence.
- Correlated variables may divide importance.
- SHAP and coefficients explain the underlying Logistic Regression score; isotonic calibration is separate.
- The final 294-row holdout is not reused for explainability.
- No post-holdout model, threshold, calibration, or preprocessing changes are made from these findings.

## Recruiter takeaway

This project does not stop at a headline accuracy number. It combines leakage-safe model development, frozen final evaluation, calibration and thresholding, and multiple complementary explanation methods so both global model behavior and individual predictions can be inspected reproducibly.
"""


def save_explainability_summary(
    summary: dict[str, object],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Persist consensus evidence and recruiter-facing Markdown."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "explainability_consensus.csv"
    json_path = output_dir / "explainability_summary.json"
    markdown_path = output_dir / "explainability_recruiter_report.md"

    summary["consensus_table"].to_csv(csv_path, index=False)
    serializable = {
        key: value for key, value in summary.items() if key != "consensus_table"
    }
    json_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_recruiter_report(summary), encoding="utf-8")

    return {
        "consensus_csv": str(csv_path),
        "summary_json": str(json_path),
        "recruiter_report": str(markdown_path),
    }


def main() -> None:
    """Run the Stage 7 explainability synthesis without using the final holdout."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )

    summary = build_explainability_summary(
        split.X_train,
        split.y_train,
        random_seed=settings.random_seed,
    )
    artifacts = save_explainability_summary(summary)

    payload = {
        key: value
        for key, value in summary.items()
        if key != "consensus_table"
    }
    payload["artifacts"] = artifacts
    payload["holdout_used_for_explainability"] = False
    payload["holdout_rows_excluded"] = len(split.X_test)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
