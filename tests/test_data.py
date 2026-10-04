"""Tests for the dataset contract."""

from pathlib import Path

import pandas as pd
import pytest

from attrition.data import (
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def source_df() -> pd.DataFrame:
    return load_dataset(DATASET)


def test_source_dataset_passes_contract(source_df: pd.DataFrame) -> None:
    audit = validate_dataset(source_df)

    assert audit.rows == 1470
    assert audit.columns == 35
    assert audit.missing_values == 0
    assert audit.duplicate_rows == 0
    assert audit.unique_employee_ids == 1470
    assert audit.attrition_yes == 237
    assert audit.attrition_no == 1233
    assert audit.attrition_rate == pytest.approx(237 / 1470)


def test_validation_rejects_duplicate_employee_id(source_df: pd.DataFrame) -> None:
    broken = source_df.copy()
    broken.loc[1, "EmployeeNumber"] = broken.loc[0, "EmployeeNumber"]

    with pytest.raises(DatasetValidationError, match="EmployeeNumber must be unique"):
        validate_dataset(broken)


def test_validation_rejects_missing_value(source_df: pd.DataFrame) -> None:
    broken = source_df.copy()
    broken.loc[0, "Age"] = None

    with pytest.raises(DatasetValidationError, match="missing values"):
        validate_dataset(broken)


def test_validation_rejects_unexpected_target(source_df: pd.DataFrame) -> None:
    broken = source_df.copy()
    broken.loc[0, "Attrition"] = "Unknown"

    with pytest.raises(DatasetValidationError, match="expected Attrition values"):
        validate_dataset(broken)
