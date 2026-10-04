"""Reusable exploratory analysis and business risk segmentation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from attrition.config import get_settings
from attrition.data import TARGET_COLUMN, load_dataset, validate_dataset

DEFAULT_SEGMENTS = (
    "OverTime",
    "JobRole",
    "Department",
    "BusinessTravel",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
)


@dataclass(frozen=True)
class AttritionSummary:
    """Overall attrition counts and rate."""

    employees: int
    attrition_yes: int
    attrition_no: int
    attrition_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def overall_attrition(df: pd.DataFrame) -> AttritionSummary:
    """Calculate the overall observed attrition rate."""

    counts = df[TARGET_COLUMN].value_counts()
    yes = int(counts.get("Yes", 0))
    no = int(counts.get("No", 0))
    total = len(df)
    return AttritionSummary(total, yes, no, yes / total)


def segment_attrition(
    df: pd.DataFrame,
    column: str,
    *,
    min_group_size: int = 20,
) -> pd.DataFrame:
    """Return descriptive attrition metrics for sufficiently large groups."""

    if column not in df.columns:
        raise ValueError(f"unknown segmentation column: {column}")
    if column == TARGET_COLUMN:
        raise ValueError("cannot segment Attrition by itself")
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1")

    observed = df.assign(_attrition=df[TARGET_COLUMN].eq("Yes").astype("int8"))
    result = (
        observed.groupby(column, dropna=False, observed=True)["_attrition"]
        .agg(employees="size", attrition_yes="sum", attrition_rate="mean")
        .reset_index()
    )
    result["attrition_no"] = result["employees"] - result["attrition_yes"]
    result = result.loc[result["employees"] >= min_group_size].copy()
    result["attrition_rate"] = result["attrition_rate"].round(6)

    return result.sort_values(
        ["attrition_rate", "employees"],
        ascending=[False, False],
    ).reset_index(drop=True)


def numeric_attrition_profile(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Compare numeric feature medians for employees who stayed vs left."""

    unknown = [column for column in columns if column not in df.columns]
    if unknown:
        raise ValueError(f"unknown numeric columns: {unknown}")

    non_numeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"non-numeric columns requested: {non_numeric}")

    medians = df.groupby(TARGET_COLUMN, observed=True)[list(columns)].median().T
    medians.index.name = "feature"
    return medians.reset_index()


def build_eda_report(df: pd.DataFrame, *, min_group_size: int = 20) -> dict[str, object]:
    """Build machine-readable descriptive EDA for later reporting/dashboard use."""

    summary = overall_attrition(df)
    segments = {
        column: segment_attrition(df, column, min_group_size=min_group_size).to_dict(
            orient="records"
        )
        for column in DEFAULT_SEGMENTS
    }
    numeric = numeric_attrition_profile(
        df,
        (
            "Age",
            "MonthlyIncome",
            "DistanceFromHome",
            "TotalWorkingYears",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "YearsSinceLastPromotion",
        ),
    ).to_dict(orient="records")

    return {
        "summary": summary.to_dict(),
        "segments": segments,
        "numeric_medians_by_attrition": numeric,
        "interpretation": (
            "Descriptive associations only; this dataset does not establish causal effects."
        ),
    }


def main() -> None:
    """Validate the source data and print the reproducible EDA report as JSON."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    print(json.dumps(build_eda_report(df), indent=2, default=str))


if __name__ == "__main__":
    main()
