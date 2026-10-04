# Exploratory analysis protocol

Stage 2.1 adds reproducible descriptive analysis without training a model.

## Purpose

The EDA layer answers two types of questions:

1. What is the observed overall attrition rate?
2. Which workforce segments show higher or lower observed attrition rates in this dataset?

The implementation is reusable Python code rather than notebook-only calculations so the same definitions can later power documentation and the Streamlit dashboard.

## Business segmentation

The first analysis set covers:

- overtime;
- job role;
- department;
- business travel;
- job satisfaction;
- environment satisfaction;
- work-life balance.

A configurable minimum group size defaults to 20 employees. This avoids highlighting tiny groups as if their observed rate were stable evidence.

The report also compares medians for selected numeric features including age, monthly income, distance from home, total working years, company tenure, years in current role, and years since last promotion.

## Interpretation rule

These results are descriptive associations in a benchmark/synthetic HR dataset. They must not be presented as proof that a factor causes employee attrition.

The EDA code does not fit encoders, scalers, resamplers, feature selectors, or predictive models and therefore does not alter the Stage 1 modeling holdout.
