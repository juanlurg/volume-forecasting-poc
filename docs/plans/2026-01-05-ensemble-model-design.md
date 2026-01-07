# Ensemble Model Design: Weighted Average of XGBoost, LightGBM, and Prophet

**Date:** 2026-01-05
**Goal:** Improve from 9.25% MAPE to < 9.0% MAPE using model ensembling

## Summary

Create an EnsembleModel class that combines predictions from XGBoost_Tuned, LightGBM_Enhanced, and Prophet_Enhanced using inverse-MAE weighted averaging.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Ensemble method | Weighted average | Simple, interpretable, effective |
| Models | XGBoost + LightGBM + Prophet (enhanced) | Diverse models, all use full feature set |
| Weight calculation | Inverse-MAE | Automatic, adapts to performance |
| Implementation | New EnsembleModel class | Reusable, fits BaseModel interface |

## Class Design

```python
class EnsembleModel(BaseModel):
    def __init__(
        self,
        models: list[BaseModel],
        weighting: str = "inverse_mae",
        name: str = "ensemble"
    )
```

## Fit Logic

1. Fit each base model on training data
2. Generate in-sample predictions from each model
3. Calculate MAE for each model
4. Compute weights: `weight_i = (1/MAE_i) / sum(1/MAE_j)`

## Predict Logic

1. Get predictions from each base model
2. Combine: `prediction = sum(weight_i * prediction_i)`
3. Return DataFrame with date and prediction columns

## Weight Calculation Example

```
Model MAEs: XGBoost=4500, LightGBM=5000, Prophet=5500
Inverse: 1/4500=0.000222, 1/5000=0.000200, 1/5500=0.000182
Normalized weights: 0.37, 0.33, 0.30
```

## Deliverables

1. `src/volume_forecast/models/ensemble.py`
2. Tests in `tests/unit/test_models.py`
3. Updated notebook 07 with ensemble comparison

## Expected Outcome

- Current: 9.25% MAPE (XGBoost_Tuned)
- Target: < 9.0% MAPE (Ensemble)
