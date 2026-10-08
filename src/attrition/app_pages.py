"""Streamlit page renderers for the attrition analytics application."""

from pathlib import Path

import pandas as pd
import streamlit as st

from attrition.app_runtime import (
    audit_as_dict,
    build_training_split,
    fit_frozen_prediction_model,
    load_validated_dataset,
    methodology_snapshot,
    overview_snapshot,
)
from attrition.config import get_settings


@st.cache_data(show_spinner=False)
def cached_dataset(path_value: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Cache validated source data for interactive analytics."""

    df, audit = load_validated_dataset(Path(path_value))
    return df, audit_as_dict(audit)


@st.cache_resource(show_spinner="Loading frozen prediction model...")
def cached_prediction_model(
    path_value: str,
    test_size: float,
    random_seed: int,
):
    """Cache the fitted frozen estimator for later prediction pages."""

    df, _ = load_validated_dataset(Path(path_value))
    split = build_training_split(
        df,
        test_size=test_size,
        random_seed=random_seed,
    )
    return fit_frozen_prediction_model(split, random_seed=random_seed)


def _load_app_data() -> tuple[pd.DataFrame, dict[str, object]]:
    settings = get_settings()
    return cached_dataset(str(settings.dataset_path))


def render_overview() -> None:
    """Render the application landing page."""

    _, audit = _load_app_data()
    snapshot = overview_snapshot_from_mapping(audit)

    st.title("Employee Attrition Analytics")
    st.caption(
        "A leakage-safe machine learning workflow for workforce analytics, "
        "calibrated attrition-risk estimation, and interpretable predictions."
    )

    cols = st.columns(4)
    cols[0].metric("Employees", f"{snapshot['employees']:,}")
    cols[1].metric("Attrition cases", f"{snapshot['attrition_cases']:,}")
    cols[2].metric("Attrition rate", f"{snapshot['attrition_rate']:.1%}")
    cols[3].metric("Final threshold", "0.38")

    st.subheader("What this application is built on")
    st.markdown(
        """
- Deterministic stratified training/holdout split.
- Fold-local preprocessing and leakage-safe model validation.
- Tuned Logistic Regression with isotonic probability calibration.
- Frozen decision threshold selected from training-only out-of-fold predictions.
- Coefficient, permutation-importance, and SHAP explanation layers.
        """
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Data quality")
        st.write(
            {
                "source columns": snapshot["source_features"],
                "missing values": snapshot["missing_values"],
                "duplicate rows": snapshot["duplicate_rows"],
            }
        )
    with right:
        st.subheader("Application modules")
        st.markdown(
            """
**Available now**
- Project overview
- Model methodology and final evaluation summary

**Next**
- Workforce analytics dashboard
- Interactive employee risk prediction
- Local SHAP explanation inside the prediction workflow
            """
        )

    st.info(
        "This project uses a benchmark HR dataset. Model outputs are analytical "
        "demonstrations and must not be used as automated employment decisions."
    )


def overview_snapshot_from_mapping(audit: dict[str, object]) -> dict[str, object]:
    """Build an overview snapshot from cached audit data."""

    return {
        "employees": int(audit["rows"]),
        "source_features": int(audit["columns"]),
        "attrition_cases": int(audit["attrition_yes"]),
        "non_attrition_cases": int(audit["attrition_no"]),
        "attrition_rate": float(audit["attrition_rate"]),
        "missing_values": int(audit["missing_values"]),
        "duplicate_rows": int(audit["duplicate_rows"]),
    }


def render_methodology() -> None:
    """Render frozen model and evaluation methodology."""

    snapshot = methodology_snapshot()
    policy = snapshot["model_policy"]
    metrics = snapshot["holdout_metrics"]
    matrix = snapshot["confusion_matrix"]

    st.title("Model & Methodology")
    st.caption("Frozen model policy and the evaluation boundary used by the project.")

    st.subheader("Frozen prediction policy")
    st.dataframe(
        pd.DataFrame(
            {
                "Component": [
                    "Model",
                    "Regularization",
                    "C",
                    "Solver",
                    "Calibration",
                    "Calibration CV",
                    "Decision threshold",
                ],
                "Configuration": [
                    policy["model"],
                    policy["regularization"],
                    policy["C"],
                    policy["solver"],
                    policy["calibration"],
                    policy["calibration_cv"],
                    policy["threshold"],
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Final holdout results")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
    metric_cols[1].metric("Precision", f"{metrics['precision']:.2%}")
    metric_cols[2].metric("Recall", f"{metrics['recall']:.2%}")
    metric_cols[3].metric("F1", f"{metrics['f1']:.2%}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
    metric_cols[1].metric("Average Precision", f"{metrics['average_precision']:.2%}")
    metric_cols[2].metric("Brier score", f"{metrics['brier_score']:.4f}")
    metric_cols[3].metric("Log loss", f"{metrics['log_loss']:.4f}")

    st.subheader("Confusion matrix")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Actual class": "No attrition",
                    "Predicted No": matrix["true_negative"],
                    "Predicted Yes": matrix["false_positive"],
                },
                {
                    "Actual class": "Attrition",
                    "Predicted No": matrix["false_negative"],
                    "Predicted Yes": matrix["true_positive"],
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Evaluation boundary")
    st.write(snapshot["evaluation_policy"])

    st.subheader("Explainability stack")
    st.markdown(
        """
- **Coefficients:** signed Logistic Regression terms and odds ratios.
- **Permutation importance:** predictive reliance on original input features.
- **Global SHAP:** model-wide contribution magnitude and direction.
- **Local SHAP:** row-level risk-increasing and risk-reducing contributions.
- **Consensus report:** agreement across independent global explanation methods.
        """
    )

    st.warning(
        "Performance values shown here are stored final results. Opening the app does "
        "not reevaluate the consumed holdout or alter the frozen model policy."
    )
