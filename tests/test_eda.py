"""Tests for reproducible exploratory analysis."""

from pathlib import Path

import pandas as pd
import pytest

from attrition.data import load_dataset
from attrition.eda import (
    build_eda_report,
    numeric_attrition_profile,
    overall_attrition,
    segment_attrition,
)

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def source_df() -> pd.DataFrame:
    return load_dataset(DATASET)


def test_overall_attrition_matches_validated_dataset(source_df: pd.DataFrame) -> None:
    summary = overall_attrition(source_df)

    assert summary.employees == 1470
    assert summary.attrition_yes == 237
    assert summary.attrition_no == 1233
    assert summary.attrition_rate == pytest.approx(237 / 1470)


def test_overtime_segmentation_has_correct_counts(source_df: pd.DataFrame) -> None:
    result = segment_attrition(source_df, "OverTime").set_index("OverTime")

    assert int(result.loc["Yes", "employees"]) == 416
    assert int(result.loc["Yes", "attrition_yes"]) == 127
    assert float(result.loc["Yes", "attrition_rate"]) == pytest.approx(127 / 416, abs=1e-6)
    assert int(result.loc["No", "employees"]) == 1054
    assert int(result.loc["No", "attrition_yes"]) == 110


def test_segment_counts_reconcile_to_total(source_df: pd.DataFrame) -> None:
    result = segment_attrition(source_df, "Department")

    assert int(result["employees"].sum()) == len(source_df)
    assert int(result["attrition_yes"].sum()) == 237
    assert int(result["attrition_no"].sum()) == 1233


def test_minimum_group_size_filters_small_segments() -> None:
    df = pd.DataFrame(
        {
            "Attrition": ["Yes", "No", "No", "Yes"],
            "Group": ["small", "large", "large", "large"],
        }
    )

    result = segment_attrition(df, "Group", min_group_size=2)

    assert result["Group"].tolist() == ["large"]
    assert int(result.loc[0, "employees"]) == 3


def test_invalid_segment_requests_are_rejected(source_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown segmentation column"):
        segment_attrition(source_df, "DoesNotExist")

    with pytest.raises(ValueError, match="cannot segment Attrition"):
        segment_attrition(source_df, "Attrition")

    with pytest.raises(ValueError, match="min_group_size"):
        segment_attrition(source_df, "Department", min_group_size=0)


def test_numeric_profile_rejects_categorical_column(source_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        numeric_attrition_profile(source_df, ("Age", "Department"))


def test_eda_report_is_machine_readable_and_labeled_descriptive(
    source_df: pd.DataFrame,
) -> None:
    report = build_eda_report(source_df)

    assert report["summary"]["employees"] == 1470
    assert "OverTime" in report["segments"]
    assert len(report["numeric_medians_by_attrition"]) == 7
    assert "causal" in report["interpretation"].lower()
