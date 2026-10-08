# Streamlit application

The repository includes a Streamlit application entrypoint at:

```text
streamlit_app.py
```

The application layer is intentionally built on top of the existing validated Python modules instead of reimplementing the ML workflow inside UI scripts.

## Current application foundation

The first application increment provides:

- a multipage shell using `st.Page` and `st.navigation`;
- an Overview page backed by the validated dataset contract;
- a Model & Methodology page showing the frozen model policy and stored final results;
- cached dataset loading with `st.cache_data`;
- cached frozen-model fitting with `st.cache_resource` for upcoming inference pages;
- reusable non-UI application services in `src/attrition/app_runtime.py`.

The application does not rescore the consumed final holdout when it starts.

## Run locally

Install the project dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Start the app:

```powershell
streamlit run streamlit_app.py
```

## Application boundaries

The frozen deployment model is trained on the same 1,176-row training partition used by the validated project workflow. The consumed 294-row final holdout is not folded back into model fitting.

The application is a benchmark analytics demonstration and must not be used for automated employment decisions.

## Next application increments

- workforce analytics and interactive filtering;
- employee risk input form;
- calibrated prediction output;
- local SHAP explanation for submitted inputs;
- application-level validation, caching review, and deployment readiness.
