"""Tests for reproducible EDA figure generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from attrition.data import load_dataset
from attrition.visualization import generate_eda_charts

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def source_df() -> pd.DataFrame:
    return load_dataset(DATASET)


def test_generate_eda_charts_creates_expected_pngs(
    source_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    paths = generate_eda_charts(source_df, tmp_path)

    assert len(paths) == 5
    assert {path.name for path in paths} == {
        "attrition_by_overtime.png",
        "attrition_by_job_role.png",
        "attrition_by_business_travel.png",
        "attrition_by_environment_satisfaction.png",
        "attrition_by_work_life_balance.png",
    }
    assert all(path.is_file() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_chart_generation_closes_figures(
    source_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    before = set(plt.get_fignums())
    generate_eda_charts(source_df, tmp_path)
    after = set(plt.get_fignums())

    assert after == before
