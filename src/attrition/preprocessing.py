"""Leakage-safe preprocessing for model pipelines."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrition.features import ENGINEERED_FEATURES, add_business_features


@dataclass(frozen=True)
class FeatureGroups:
    """Column names assigned to numeric and categorical preprocessing."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


class BusinessFeatureTransformer(TransformerMixin, BaseEstimator):
    """Scikit-learn-compatible deterministic business feature transformer."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "BusinessFeatureTransformer":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_business_features(X)

    def get_feature_names_out(
        self,
        input_features: object = None,
    ) -> np.ndarray:
        """Return original plus deterministic engineered feature names."""

        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray([*input_features, *ENGINEERED_FEATURES], dtype=object)


def infer_feature_groups(X: pd.DataFrame) -> FeatureGroups:
    """Infer numeric and categorical columns from engineered model inputs."""

    engineered = add_business_features(X)
    categorical = tuple(engineered.select_dtypes(include=["object", "category"]).columns)
    numeric = tuple(column for column in engineered.columns if column not in categorical)

    if not numeric or not categorical:
        raise ValueError("preprocessing requires both numeric and categorical features")

    return FeatureGroups(numeric=numeric, categorical=categorical)


def build_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """Build an unfitted feature-engineering and preprocessing pipeline."""

    groups = infer_feature_groups(X_train)
    columns = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(groups.numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(groups.categorical),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("business_features", BusinessFeatureTransformer()),
            ("columns", columns),
        ]
    )
