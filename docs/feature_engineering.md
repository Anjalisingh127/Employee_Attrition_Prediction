# Stage 3.1 — business feature engineering

Feature engineering is intentionally small, deterministic, and independent of the target. The goal is to expose interpretable relationships without creating a large collection of speculative features.

## Engineered features

| Feature | Definition | Rationale |
| --- | --- | --- |
| `IncomePerJobLevel` | MonthlyIncome / JobLevel | Normalizes compensation by organizational level. |
| `CompanyTenureRatio` | YearsAtCompany / TotalWorkingYears | Describes how much of a career has been spent at the current company. |
| `RoleTenureRatio` | YearsInCurrentRole / YearsAtCompany | Describes stability within the current role relative to company tenure. |
| `PromotionWaitRatio` | YearsSinceLastPromotion / YearsAtCompany | Describes promotion recency relative to company tenure. |
| `EarlyCareer` | TotalWorkingYears < 5 | Provides a simple career-stage indicator. |

Zero denominators are mapped to 0 for ratio features so the transformation never creates infinity or missing values.

## Leakage rule

No engineered feature uses `Attrition`, class frequencies, target means, fitted statistics, or information from other employees. The transformation is therefore deterministic at the individual-row level and can be applied consistently to training, test, and future inference records.

These features are candidates, not claims that they improve prediction. Their value must be established by cross-validation during model development.
