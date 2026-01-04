# Volume Forecasting POC - Design Document

**Date**: 2026-01-04
**Status**: Approved
**Scope**: Daily login and deposit volume forecasting for betting platform

---

## Overview

A proof-of-concept for forecasting daily logins and deposits on a UK betting company platform. The system considers sports events (football, horse racing, tennis, boxing/UFC), UK bank holidays, and typical behavioral patterns.

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Forecast horizon | 7 days |
| Target metrics | Daily logins, daily deposits (count + volume) |
| Validation approach | Walk-forward temporal validation |
| Deliverables | Python package + exploration notebooks |

### Explicit Exclusions (YAGNI)

- Real-time predictions (batch daily only)
- User-level modeling (aggregate volumes only)
- Automated retraining / MLOps pipeline

---

## Project Structure

```
volume-forecasting-poc/
├── pyproject.toml              # uv/pip project config
├── README.md                   # Project overview & quickstart
├── src/
│   └── volume_forecast/
│       ├── __init__.py
│       ├── data_generation/    # Synthetic data creation
│       ├── external_data/      # API clients (sports, holidays)
│       ├── features/           # Feature engineering pipelines
│       ├── models/             # Model implementations
│       ├── evaluation/         # Metrics & validation
│       └── config/             # Settings, constants
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_external_data.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_statistical_models.ipynb
│   ├── 06_ml_models.ipynb
│   ├── 07_model_comparison.ipynb
│   └── 08_final_forecast.ipynb
├── data/
│   ├── raw/                    # Generated synthetic data
│   ├── processed/              # Feature-engineered data
│   └── external/               # Cached API responses
├── tests/
├── docs/
│   └── plans/
└── .gitignore
```

---

## Synthetic Data Generation

Simulates ~2 years of realistic betting platform behavior for train/test/validation splits.

### Module Structure

```
data_generation/
├── __init__.py
├── base.py              # Base generator class, random seed management
├── calendar.py          # Time patterns (day-of-week, month, UK holidays)
├── events.py            # Sports event impact simulation
├── user_segments.py     # Casual, regular, high-roller behavior profiles
├── correlations.py      # Login→deposit relationships, churn effects
├── promotions.py        # Promotional campaign impact simulation
└── generator.py         # Main orchestrator combining all components
```

### Generated Data

- `daily_logins` - Total platform logins per day
- `daily_deposits` - Total deposit count and volume (GBP)
- Underlying factors stored for validation (event flags, promo flags, etc.)

### Realism Patterns

- **Weekly cycle**: Weekend peaks (especially Saturday), Monday dip
- **Seasonal**: Football season (Aug-May) vs. summer lull, Christmas spike
- **Event-driven**: 2-5x multipliers around major matches/races
- **Correlations**: ~70% of depositors logged in same day; deposits lag logins by hours
- **Noise**: Realistic variance, occasional anomalies

### Output Format

CSV files + metadata JSON describing generation parameters for reproducibility.

---

## External Data Integration

Real API integrations for sports events and UK holidays with caching and fallback handling.

### Module Structure

```
external_data/
├── __init__.py
├── base.py              # Base API client with retry, caching, rate limiting
├── holidays.py          # UK Government Bank Holidays API
├── football.py          # football-data.org API (free tier)
├── racing.py            # Horse racing calendar
├── tennis.py            # Grand Slam dates
├── boxing_ufc.py        # Major fight calendar
└── aggregator.py        # Unified event calendar interface
```

### Data Sources

| Data | Source | Method |
|------|--------|--------|
| UK Bank Holidays | gov.uk/bank-holidays.json | Free, no auth |
| Football (PL, CL, Euros, World Cup) | football-data.org | Free tier, API key |
| Horse Racing (major meets) | Static JSON + API | Curated + API |
| Tennis Grand Slams | Static JSON | 4 events/year |
| Boxing/UFC major fights | Static JSON | Curated notable events |

### Unified Event Interface

```python
{
    "date": "2024-05-25",
    "event_type": "football",
    "importance": "high",  # low, medium, high, major
    "name": "Champions League Final"
}
```

### Design Decisions

- All API responses cached locally in `data/external/` with TTL
- Graceful fallback to cached data if API unavailable
- Importance levels for feature engineering weighting

---

## Feature Engineering

Transform raw data + external events into model-ready features.

### Module Structure

```
features/
├── __init__.py
├── base.py              # Base transformer interface
├── temporal.py          # Time-based features
├── events.py            # Sports event features
├── lags.py              # Autoregressive lag features
├── rolling.py           # Rolling window statistics
├── holidays.py          # UK holiday features
└── pipeline.py          # Full feature pipeline orchestrator
```

### Feature Categories

| Category | Features |
|----------|----------|
| **Temporal** | day_of_week, month, is_weekend, week_of_year, is_month_start/end, days_to_payday (25th) |
| **Holiday** | is_uk_holiday, days_to_next_holiday, holiday_type (bank, christmas, easter) |
| **Events** | event_count_today, max_event_importance, is_football_matchday, is_major_race, days_since_last_major |
| **Lag** | logins_lag_1d, logins_lag_7d, deposits_lag_1d, deposits_lag_7d |
| **Rolling** | logins_7d_mean, logins_7d_std, deposits_30d_trend, week_over_week_change |
| **Cyclical** | sin/cos transforms for day_of_week, month |

### Design Decisions

- All transformers are sklearn-compatible (`fit`/`transform`)
- Pipeline outputs single DataFrame with original target + all features
- Feature names are self-documenting for interpretability
- Lag features handle missing values at series start gracefully

---

## Model Architecture

Systematic comparison across model families, from simple baselines to advanced ML.

### Module Structure

```
models/
├── __init__.py
├── base.py              # Abstract base model interface
├── baselines.py         # Naive, seasonal naive, moving average
├── statistical.py       # ARIMA, SARIMA, ETS
├── prophet_model.py     # Facebook Prophet with regressors
├── tree_models.py       # XGBoost, LightGBM, Random Forest
├── linear.py            # Ridge, Lasso, ElasticNet
├── gam.py               # Generalized Additive Models
├── ensemble.py          # Stacking, weighted averaging
└── registry.py          # Model registry for easy instantiation
```

### Model Progression (Simple → Complex)

| Tier | Models | Purpose |
|------|--------|---------|
| **Baselines** | Naive (yesterday), Seasonal Naive (last week), 7-day MA | Minimum performance bar |
| **Statistical** | SARIMA, ETS | Classic time series approaches |
| **Prophet** | Prophet + event/holiday regressors | Handles seasonality + external factors |
| **Linear ML** | Ridge, ElasticNet | Interpretable ML baseline |
| **GAMs** | pyGAM | Interpretable non-linear relationships |
| **Tree-based** | XGBoost, LightGBM | Strong feature-based performance |
| **Ensemble** | Weighted average of top performers | Final production candidate |

### Unified Interface

```python
class BaseModel(ABC):
    def fit(self, train_df: pd.DataFrame, target: str) -> "BaseModel"
    def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame
    def get_params(self) -> dict
```

---

## Evaluation Framework

Temporal validation with walk-forward approach and comprehensive metrics.

### Module Structure

```
evaluation/
├── __init__.py
├── metrics.py           # MAPE, RMSE, MAE, custom metrics
├── validation.py        # Walk-forward, expanding window splitters
├── comparison.py        # Model comparison tables and rankings
└── visualization.py     # Forecast plots, residual analysis
```

### Walk-Forward Validation

```
|-------- Training --------|-- Test --|
                           |--- 7d ---|

Fold 1: [Day 1 -------- Day 365] → predict [Day 366-372]
Fold 2: [Day 1 -------- Day 372] → predict [Day 373-379]
Fold 3: [Day 1 -------- Day 379] → predict [Day 380-386]
...
```

- **Minimum training window**: 365 days (capture full seasonality)
- **Test window**: 7 days (matches forecast horizon)
- **Step size**: 7 days (non-overlapping test periods)
- **Expanding window**: Training grows each fold

### Metrics

| Metric | Purpose |
|--------|---------|
| MAE | Average absolute error in original units |
| RMSE | Penalizes large errors more heavily |
| MAPE | Percentage error for business interpretation |
| sMAPE | Symmetric MAPE, handles zeros better |
| Coverage | % of actuals within prediction intervals |

### Output

Comparison DataFrame with mean/std metrics per model across all folds, plus per-fold breakdown for stability analysis.

---

## Notebooks & Documentation

### Notebook Flow

| Notebook | Purpose |
|----------|---------|
| 01_data_exploration.ipynb | Explore synthetic data, validate realism |
| 02_external_data.ipynb | Fetch & visualize sports/holiday calendars |
| 03_feature_engineering.ipynb | Feature analysis, correlation heatmaps |
| 04_baseline_models.ipynb | Naive, MA, seasonal naive benchmarks |
| 05_statistical_models.ipynb | SARIMA, ETS, Prophet |
| 06_ml_models.ipynb | XGBoost, LightGBM, GAMs |
| 07_model_comparison.ipynb | Head-to-head evaluation, winner selection |
| 08_final_forecast.ipynb | Production-style forecast with best model |

### Documentation Files

- `docs/data_dictionary.md` - Field definitions for synthetic data
- `docs/model_catalog.md` - Description of each model, hyperparams
- `docs/api_reference.md` - External API docs and rate limits
- `README.md` - Project overview, quickstart, notebook guide

---

## Technical Stack

- **Python**: 3.11+
- **Package manager**: uv
- **Core libraries**: pandas, numpy, scikit-learn
- **Time series**: statsmodels, prophet
- **ML**: xgboost, lightgbm, pygam
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: jupyter
- **Testing**: pytest

---

## Next Steps

1. Research phase (subagent-based):
   - Model architectures for demand forecasting
   - Betting industry domain patterns
   - Python/uv project setup best practices

2. Implementation via `superpowers:writing-plans` skill

3. Development following TDD practices
