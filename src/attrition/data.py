"""Dataset loading and structural validation."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Attrition"
ID_COLUMN = "EmployeeNumber"
EXPECTED_ROWS = 1470
EXPECTED_COLUMNS = 35
EXPECTED_TARGET_VALUES = {"Yes", "No"}
INVARIANT_COLUMNS = ("EmployeeCount", "Over18", "StandardHours")


class DatasetValidationError(ValueError):
    """Raised when the source dataset violates a required invariant."""


@dataclass(frozen=True)
class DatasetAudit:
    """Machine-readable summary of the source dataset."""

    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    unique_employee_ids: int
    attrition_yes: int
    attrition_no: int
    attrition_rate: float
    invariant_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the audit for CLI/reporting use."""

        return asdict(self)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the source CSV without mutating its values."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> DatasetAudit:
    """Validate the known IBM HR dataset contract and return an audit."""

    errors: list[str] = []

    if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        errors.append(
            f"expected shape {(EXPECTED_ROWS, EXPECTED_COLUMNS)}, found {df.shape}"
        )

    required = {TARGET_COLUMN, ID_COLUMN, *INVARIANT_COLUMNS}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")

    if errors:
        raise DatasetValidationError("; ".join(errors))

    target_values = set(df[TARGET_COLUMN].dropna().unique())
    if target_values != EXPECTED_TARGET_VALUES:
        errors.append(
            f"expected Attrition values {sorted(EXPECTED_TARGET_VALUES)}, "
            f"found {sorted(target_values)}"
        )

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    unique_employee_ids = int(df[ID_COLUMN].nunique(dropna=False))

    if missing_values:
        errors.append(f"found {missing_values} missing values")
    if duplicate_rows:
        errors.append(f"found {duplicate_rows} duplicate rows")
    if unique_employee_ids != len(df):
        errors.append(
            f"EmployeeNumber must be unique: {unique_employee_ids} unique IDs for {len(df)} rows"
        )

    invariant_columns = tuple(
        column for column in INVARIANT_COLUMNS if df[column].nunique(dropna=False) == 1
    )
    if set(invariant_columns) != set(INVARIANT_COLUMNS):
        errors.append(
            "expected invariant columns to remain constant: "
            f"{list(INVARIANT_COLUMNS)}"
        )

    if errors:
        raise DatasetValidationError("; ".join(errors))

    counts = df[TARGET_COLUMN].value_counts()
    yes_count = int(counts.get("Yes", 0))
    no_count = int(counts.get("No", 0))

    return DatasetAudit(
        rows=len(df),
        columns=len(df.columns),
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        unique_employee_ids=unique_employee_ids,
        attrition_yes=yes_count,
        attrition_no=no_count,
        attrition_rate=yes_count / len(df),
        invariant_columns=invariant_columns,
    )
