# Full Hyperparameter Tuning Design: LightGBM + Prophet + Tuned Ensemble

**Date:** 2026-01-05
**Goal:** Tune all ensemble component models to break below 9.0% MAPE

## Current State

| Model | MAPE | Status |
|-------|------|--------|
| XGBoost_Tuned | 9.48% | Tuned (50 trials) |
| LightGBM_Enhanced | 10.74% | Default params |
| Prophet_Enhanced | 10.71% | Default params |
| Current Ensemble | 9.44% | 1 tuned + 2 default |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tuning scope | All models | Both LightGBM and Prophet have room to improve |
| Trials per model | 50 | Good balance of exploration vs runtime |
| Implementation | Extend notebook 08 | Keep all tuning in one place |
| LightGBM params | Core only | n_estimators, max_depth, learning_rate, num_leaves |
| Prophet params | Seasonality + changepoints | Both affect Prophet performance significantly |

## LightGBM Search Space

```python
params = {
    'n_estimators': (50, 500),        # Integer, uniform
    'max_depth': (3, 15),             # Integer, uniform
    'learning_rate': (0.01, 0.3),     # Float, log-uniform
    'num_leaves': (15, 127),          # Integer, uniform
}
```

## Prophet Search Space

```python
params = {
    # Seasonality
    'seasonality_prior_scale': (0.01, 10.0),    # Float, log-uniform
    'seasonality_mode': ['additive', 'multiplicative'],  # Categorical

    # Changepoints
    'changepoint_prior_scale': (0.001, 0.5),    # Float, log-uniform
    'n_changepoints': (10, 50),                  # Integer, uniform
    'changepoint_range': (0.7, 0.95),           # Float, uniform
}
```

## Deliverables

1. Update `notebooks/08_hyperparameter_tuning.ipynb` with LightGBM and Prophet tuning
2. Save best params to JSON files
3. Update `notebooks/07_model_comparison.ipynb` with Ensemble_Tuned
4. Run final comparison

## Expected Outcome

- Current: 9.44% MAPE (Ensemble with 1 tuned model)
- Target: < 9.0% MAPE (Ensemble with 3 tuned models)

## Runtime Estimate

- LightGBM tuning: ~15-20 minutes
- Prophet tuning: ~20-25 minutes
- Total: ~35-45 minutes
