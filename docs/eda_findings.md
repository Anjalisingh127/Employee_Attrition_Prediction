# Stage 2.2 — EDA findings

These findings are generated from the validated 1,470-row IBM HR Analytics dataset. They describe associations in a benchmark/synthetic dataset and do not establish causal relationships.

## Baseline

The dataset contains 237 employees marked as attrition cases and 1,233 marked as non-attrition cases, for an observed attrition rate of 16.12%.

## Strong descriptive patterns

### Overtime

Employees marked as working overtime show 127 attrition cases among 416 employees, an observed rate of 30.53%. Employees not working overtime show 110 cases among 1,054 employees, or 10.44%.

The overtime group therefore has an observed attrition rate about 2.9 times the non-overtime group in this dataset.

### Job role

Sales Representative has the highest observed role-level attrition rate among groups passing the minimum-size rule: 33 of 83 employees, or 39.76%.

Other comparatively high observed rates include Laboratory Technician at 23.94% and Human Resources at 23.08%. Research Director has the lowest observed rate at 2.50%.

These differences should be treated as descriptive segment patterns, not evidence that job role itself causes attrition.

### Business travel

Employees in the Travel_Frequently group show 69 attrition cases among 277 employees, or 24.91%. Travel_Rarely is 14.96%, while Non-Travel is 8.00%.

### Satisfaction and work-life balance

EnvironmentSatisfaction level 1 has a 25.35% observed attrition rate, compared with 13.45% at level 4.

WorkLifeBalance level 1 has a 31.25% observed attrition rate. Level 3, the largest group, has a 14.22% rate.

The WorkLifeBalance scale is not perfectly monotonic: level 4 has a higher observed rate than level 3. This is one reason the analysis avoids simplistic causal conclusions.

## Numeric median comparisons

Employees marked as attrition cases have lower medians for several workforce measures:

| Feature | Stayed | Left |
| --- | ---: | ---: |
| Age | 36 | 32 |
| MonthlyIncome | 5,204 | 3,202 |
| DistanceFromHome | 7 | 9 |
| TotalWorkingYears | 10 | 7 |
| YearsAtCompany | 6 | 3 |
| YearsInCurrentRole | 3 | 2 |
| YearsSinceLastPromotion | 1 | 1 |

These univariate comparisons can be influenced by correlated factors such as job level, role, age, or tenure. Predictive modeling must therefore evaluate features jointly rather than treating these medians as isolated explanations.

## What advances to modeling

Stage 2 identifies overtime, job role, business travel, satisfaction, work-life balance, income, age, experience, distance, and tenure as useful variables to examine during feature engineering and modeling.

No variable is selected or removed solely because of its EDA relationship with the target. Feature decisions and model performance will be evaluated within the leakage-safe training workflow defined in Stage 1.
