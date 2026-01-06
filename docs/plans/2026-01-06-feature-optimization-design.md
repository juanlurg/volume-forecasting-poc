# Feature Selection & Engineering Design

## Goal
Break below 9% MAPE (currently at 9.44% with Ensemble model).

## Approach
1. Use XGBoost feature importance for selection
2. Add new features from multiple categories
3. Remove bottom 20% low-importance features
4. Retrain ensemble with optimized feature set

## Notebook Structure: 09_feature_optimization.ipynb

### Section 1: Setup & Data Loading
- Load data, apply current feature pipeline
- Train XGBoost to get baseline importance

### Section 2: Feature Importance Analysis
- Extract `feature_importances_` from XGBoost
- Visualize as horizontal bar chart
- Identify bottom 20% for removal

### Section 3: New Feature Engineering
- Add new features from all three categories
- Re-run feature pipeline

### Section 4: Feature Selection
- Remove low-importance features
- Combine with new features
- Re-run importance analysis

### Section 5: Model Comparison
- Train Ensemble with optimized features
- Compare to 9.44% baseline

## New Features to Add (14 total)

### Time-based (4)
- `week_of_month`: 1-5, captures monthly patterns
- `days_to_month_end`: 0-30, more granular than is_month_end
- `days_from_month_start`: 0-30, complement to above
- `quarter`: 1-4, seasonal business patterns

### Interaction (4)
- `weekend_event`: is_weekend × has_any_event
- `payday_weekend`: days_to_payday ≤ 2 AND is_weekend
- `month_end_rolling`: is_month_end × rolling_mean_7
- `high_importance_weekend`: event_importance ≥ 3 AND is_weekend

### Advanced Lags (6)
- `lag_2`, `lag_3`: Short-term momentum
- `lag_28`: Monthly cycle (4 weeks)
- `lag_diff_7`: value - lag_7 (week-over-week change)
- `lag_diff_28`: value - lag_28 (month-over-month change)
- `lag_ratio_7`: value / lag_7 (relative weekly change)

## Selection Threshold
- Remove features with importance below 20th percentile
- Conservative approach to avoid removing useful features

## Output Artifacts
- `data/processed/feature_importance.csv` - full importance ranking
- `data/processed/selected_features.json` - optimized feature list
- Updated model comparison results

## Success Criteria
- MAPE < 9.00% (improvement from 9.44%)
