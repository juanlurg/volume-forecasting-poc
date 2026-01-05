# Event Features Integration Design

## Overview

Integrate external event data (bank holidays, sports events) into the forecasting pipeline and update models to use the full feature set. This addresses the gap where feature engineering work exists but isn't being used in model training.

## Problem

1. External data infrastructure exists (`EventAggregator`, `UKHolidaysClient`, `FootballClient`, `StaticEventsCalendar`) but isn't integrated into the feature pipeline
2. Tree models (XGBoost, LightGBM) only use `lag_1, lag_7, lag_14` - ignoring 34 other features in `features.csv`
3. Prophet uses built-in seasonality but no external regressors

## Solution

### Event Features

New columns to add to the feature pipeline:

| Feature | Type | Description |
|---------|------|-------------|
| `is_bank_holiday` | bool | UK bank holiday |
| `is_racing_event` | bool | Horse racing (Cheltenham, Ascot, etc.) |
| `is_tennis_event` | bool | Grand Slam days |
| `is_boxing_event` | bool | Major boxing/UFC |
| `is_football_match` | bool | Premier League match day |
| `event_importance` | int | 0=none, 1=low, 2=medium, 3=high, 4=major |
| `event_count` | int | Number of events on that day |

### Data Flow

```
raw_data.csv
    → FeaturePipeline (temporal, lag, rolling)
    → EventFeatures (NEW - adds 7 event columns)
    → features.csv (45 features total)
```

### Model Updates

**Tree Models (XGBoost, LightGBM):**
- Add `external_features: list[str] | None` parameter to `__init__`
- Combine lag features + external features during fitting
- Accept `future_df` in `predict()` for external feature values

**Prophet:**
- Add `regressors: list[str] | None` parameter to `__init__`
- Call `add_regressor()` for each before fitting
- Include regressor columns in future dataframe for prediction

**Unchanged:**
- Baseline models (Naive, SeasonalNaive, MA) - intentionally simple
- ARIMA/SARIMA - pure time series, don't use external features

### Benchmark Updates

1. Regenerate `features.csv` with event columns
2. Instantiate tree models with full feature set
3. Instantiate Prophet with event regressors
4. Update `WalkForwardValidator` to pass feature data per fold
5. Update `07_model_comparison.ipynb` to show improved results

## File Changes

| File | Change |
|------|--------|
| `src/volume_forecast/features/events.py` | NEW - `EventFeatures` transformer |
| `src/volume_forecast/features/__init__.py` | Export `EventFeatures` |
| `src/volume_forecast/features/pipeline.py` | Add `EventFeatures` to pipeline |
| `src/volume_forecast/models/tree_models.py` | Add `external_features` support |
| `src/volume_forecast/models/prophet_model.py` | Add `regressors` support |
| `src/volume_forecast/evaluation/validation.py` | Pass feature data in walk-forward |
| `notebooks/07_model_comparison.ipynb` | Re-run with enhanced models |
| `data/processed/features.csv` | Regenerated with event columns |

## External Dependencies

- Football data: API key stored in `.env` as `FOOTBALL_API_KEY`
- UK holidays: gov.uk API (no key required)
- Racing/Tennis/Boxing: Static calendar (no API required)

## Expected Outcome

ML models (XGBoost, LightGBM) should significantly improve relative to current benchmarks where they ranked 7th and 3rd. With access to calendar features, rolling statistics, and event indicators, they should compete more effectively with Prophet.
