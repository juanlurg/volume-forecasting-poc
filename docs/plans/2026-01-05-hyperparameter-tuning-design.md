# Hyperparameter Tuning Design: XGBoost Optimization with Optuna

**Date:** 2026-01-05
**Goal:** Improve XGBoost_Enhanced from 10.00% MAPE to < 9.0% MAPE

## Summary

Use Optuna Bayesian optimization to tune core XGBoost hyperparameters over 50 trials.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Optimization method | Optuna | Bayesian search, smart pruning, modern standard |
| Parameter scope | Core only | 80%+ of gains from n_estimators, max_depth, learning_rate, min_child_weight |
| Number of trials | 50 | Good balance of exploration vs runtime (~15-20 min) |
| Implementation | New notebook | Keeps concerns separated, interactive exploration |

## Parameter Search Space

```python
params = {
    'n_estimators': (50, 500),        # Integer, uniform
    'max_depth': (3, 12),             # Integer, uniform
    'learning_rate': (0.01, 0.3),     # Float, log-uniform
    'min_child_weight': (1, 10),      # Integer, uniform
}
```

## Evaluation Strategy

- Objective: Minimize mean MAE across 52 walk-forward folds
- Sampler: TPESampler (Tree-structured Parzen Estimator) with seed=42
- Pruner: MedianPruner with 10 startup trials

## Notebook Structure

`08_hyperparameter_tuning.ipynb`:
1. Setup - Imports, load data, apply FeaturePipeline
2. Define objective function
3. Run 50-trial optimization
4. Analyze results (history plot, parameter importance)
5. Validate best model vs default
6. Save best params to JSON

## Expected Outcomes

- Target: < 9.0% MAPE (10%+ relative improvement)
- Runtime: ~15-20 minutes
- Output: best_xgboost_params.json
