"""Deterministic business feature engineering without target leakage."""

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = (
    "IncomePerJobLevel",
    "CompanyTenureRatio",
    "RoleTenureRatio",
    "PromotionWaitRatio",
    "EarlyCareer",
)


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a small set of interpretable row-level features."""

    required = {
        "Age",
        "JobLevel",
        "MonthlyIncome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing columns required for feature engineering: {missing}")

    result = df.copy()
    result["IncomePerJobLevel"] = result["MonthlyIncome"] / result["JobLevel"]
    result["CompanyTenureRatio"] = _safe_ratio(
        result["YearsAtCompany"], result["TotalWorkingYears"]
    )
    result["RoleTenureRatio"] = _safe_ratio(
        result["YearsInCurrentRole"], result["YearsAtCompany"]
    )
    result["PromotionWaitRatio"] = _safe_ratio(
        result["YearsSinceLastPromotion"], result["YearsAtCompany"]
    )
    result["EarlyCareer"] = (result["TotalWorkingYears"] < 5).astype("int8")

    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely while keeping zero-denominator rows finite."""

    safe_denominator = denominator.replace(0, np.nan)
    return numerator.div(safe_denominator).fillna(0.0)
