# Stage 3.2 — leakage-safe preprocessing

The preprocessing layer is designed to be embedded directly inside future model pipelines.

## Pipeline order

1. Add deterministic business features.
2. Standardize numeric features with `StandardScaler`.
3. One-hot encode categorical features with `OneHotEncoder(handle_unknown="ignore")`.

Feature groups are inferred from the training feature frame after deterministic feature engineering. The reference schema contains 27 numeric and 8 categorical columns after the five Stage 3.1 features are added.

## Leakage controls

`build_preprocessor(X_train)` returns an **unfitted** pipeline. Learned statistics such as numeric means/standard deviations and categorical vocabularies are created only when `fit` or `fit_transform` is called on training data.

The final test partition is transformed only after the preprocessor has been fitted on training data.

Unknown categorical values are ignored rather than causing inference failures, which also supports later Streamlit predictions for valid categories not observed in a particular training fold.

## Modeling integration

Stage 4 models will wrap this preprocessor and the estimator in one scikit-learn pipeline. Cross-validation will therefore refit preprocessing independently inside each training fold rather than preprocessing the complete training dataset before CV.
