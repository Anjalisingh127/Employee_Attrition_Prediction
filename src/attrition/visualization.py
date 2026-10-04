"""Generate reproducible EDA charts for the attrition dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.eda import segment_attrition

DEFAULT_OUTPUT_DIR = Path("reports/generated")


def _save_rate_chart(
    data: pd.DataFrame,
    *,
    category: str,
    title: str,
    output_path: Path,
) -> Path:
    """Save a readable attrition-rate bar chart."""

    plot_data = data.sort_values("attrition_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.55 * len(plot_data))))
    ax.barh(plot_data[category].astype(str), plot_data["attrition_rate"] * 100)
    ax.set_title(title)
    ax.set_xlabel("Observed attrition rate (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    for index, value in enumerate(plot_data["attrition_rate"] * 100):
        ax.text(value + 0.5, index, f"{value:.1f}%", va="center")

    upper = max(10.0, float((plot_data["attrition_rate"] * 100).max()) * 1.2)
    ax.set_xlim(0, upper)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_eda_charts(
    df: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Generate the small recruiter-facing Stage 2.2 chart set."""

    specs = (
        ("OverTime", "Attrition rate by overtime", "attrition_by_overtime.png"),
        ("JobRole", "Attrition rate by job role", "attrition_by_job_role.png"),
        (
            "BusinessTravel",
            "Attrition rate by business travel",
            "attrition_by_business_travel.png",
        ),
        (
            "EnvironmentSatisfaction",
            "Attrition rate by environment satisfaction",
            "attrition_by_environment_satisfaction.png",
        ),
        (
            "WorkLifeBalance",
            "Attrition rate by work-life balance",
            "attrition_by_work_life_balance.png",
        ),
    )

    return [
        _save_rate_chart(
            segment_attrition(df, column),
            category=column,
            title=title,
            output_path=output_dir / filename,
        )
        for column, title, filename in specs
    ]


def main() -> None:
    """Validate the configured dataset and generate the Stage 2.2 figures."""

    settings = get_settings()
    df = load_dataset(settings.dataset_path)
    validate_dataset(df)

    for path in generate_eda_charts(df):
        print(path)


if __name__ == "__main__":
    main()
