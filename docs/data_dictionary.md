# Data dictionary

The source dataset contains 1,470 employee records and 35 columns. Stage 1 keeps the raw source unchanged and validates its structural contract before any modeling.

## Modeling treatment

| Category | Columns | Stage 1 decision |
| --- | --- | --- |
| Target | `Attrition` | Binary outcome: Yes/No |
| Identifier | `EmployeeNumber` | Validate uniqueness; exclude from model features |
| Invariant | `EmployeeCount`, `Over18`, `StandardHours` | Validate constants; exclude from model features |
| Categorical | `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime` | Encode inside the future training pipeline |
| Numeric / ordinal | Remaining columns | Preserve raw values; preprocessing will be model-specific |

## Leakage policy

No resampling, scaling, encoding, feature selection, or learned transformation is performed before the train/test split. Future transformations will be fitted on training data only and encapsulated in pipelines so cross-validation cannot learn from validation folds.

## Source-data policy

The CSV under `data/` is treated as immutable source data. Generated datasets, models, metrics, and plots will be written outside the raw-data location in later stages.
