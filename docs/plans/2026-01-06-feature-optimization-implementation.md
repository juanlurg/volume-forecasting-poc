# Feature Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create notebook 09_feature_optimization.ipynb to break below 9% MAPE through feature selection and engineering.

**Architecture:** Single Jupyter notebook with 5 sections: setup, feature importance analysis, new feature engineering, feature selection, and final model comparison. Uses XGBoost feature_importances_ for selection, adds 14 new features, removes bottom 20%.

**Tech Stack:** Python, pandas, numpy, XGBoost, LightGBM, Prophet, matplotlib, seaborn, existing FeaturePipeline

---

### Task 1: Create notebook with setup and data loading

**Files:**
- Create: `notebooks/09_feature_optimization.ipynb`

**Step 1: Create notebook with markdown header and imports cell**

Create the notebook file with initial cells:

Cell 0 (markdown):
```markdown
# 09 - Feature Selection & Engineering Optimization

This notebook optimizes features to break below 9% MAPE (currently at 9.44% with Ensemble).

## Approach
1. Analyze feature importance using XGBoost
2. Add new engineered features (time-based, interaction, advanced lags)
3. Remove bottom 20% low-importance features
4. Retrain ensemble with optimized feature set

## Target
- Current best: 9.44% MAPE (Ensemble)
- Goal: < 9.00% MAPE
```

Cell 1 (code):
```python
import sys
import json
import warnings
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

# Import models and evaluation
from volume_forecast.evaluation import WalkForwardValidator, ModelBenchmark
from volume_forecast.models import (
    XGBoostModel, LightGBMModel, ProphetModel, EnsembleModel
)
from volume_forecast.features import FeaturePipeline

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('seaborn-v0_8-whitegrid')

print("Setup complete!")
```

**Step 2: Verify notebook created correctly**

Run: `jupyter nbconvert --to notebook --execute notebooks/09_feature_optimization.ipynb --ExecutePreprocessor.timeout=60 --inplace 2>&1 | head -20`

---

### Task 2: Add data loading and baseline feature importance cells

**Files:**
- Modify: `notebooks/09_feature_optimization.ipynb`

**Step 1: Add data loading cell**

Cell 2 (code):
```python
# Load data
data_path = project_root / "data" / "raw" / "synthetic_volumes.csv"
df = pd.read_csv(data_path, parse_dates=["date"])
df = df.sort_values('date').reset_index(drop=True)

# Apply feature pipeline
pipeline = FeaturePipeline(
    date_column='date',
    target_columns=['daily_logins'],
    include_events=True,
    include_football=True,
)
df_features = pipeline.fit_transform(df)

TARGET = 'daily_logins'
FEATURE_COLUMNS = pipeline.get_feature_names()

print(f"Dataset shape: {df_features.shape}")
print(f"Features: {len(FEATURE_COLUMNS)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
```

**Step 2: Add baseline XGBoost training cell**

Cell 3 (markdown):
```markdown
## 1. Baseline Feature Importance Analysis
Train XGBoost with current features to establish importance baseline.
```

Cell 4 (code):
```python
# Define external features (same as notebook 07)
EXTERNAL_FEATURES = [
    'day_of_week', 'is_weekend', 'day_of_week_sin', 'day_of_week_cos',
    'month', 'month_sin', 'month_cos',
    'daily_logins_rolling_mean_7', 'daily_logins_rolling_mean_14', 'daily_logins_rolling_mean_30',
    'daily_logins_rolling_std_7', 'daily_logins_rolling_std_14', 'daily_logins_rolling_std_30',
    'is_bank_holiday', 'is_racing_event', 'is_tennis_event',
    'is_boxing_event', 'is_football_match', 'event_importance',
    'any_event_tomorrow', 'any_event_in_2_days', 'any_event_in_3_days',
    'bank_holiday_tomorrow', 'bank_holiday_in_2_days', 'bank_holiday_in_3_days',
    'football_tomorrow', 'football_in_2_days', 'football_in_3_days',
    'any_event_yesterday', 'any_event_2_days_ago',
    'bank_holiday_yesterday', 'bank_holiday_2_days_ago',
    'football_yesterday', 'football_2_days_ago',
]

# Train XGBoost with best params to get feature importance
xgb_model = XGBoostModel(
    n_estimators=88,
    max_depth=3,
    learning_rate=0.1105,
    min_child_weight=8,
    lags=[1, 7, 14],
    external_features=EXTERNAL_FEATURES,
    name='XGBoost_Baseline'
)

# Use first 365 days for training
train_df = df_features.iloc[:365].copy()
xgb_model.fit(train_df, TARGET)

print("XGBoost trained for feature importance analysis")
```

---

### Task 3: Add feature importance visualization

**Files:**
- Modify: `notebooks/09_feature_optimization.ipynb`

**Step 1: Add importance extraction and visualization cell**

Cell 5 (code):
```python
# Extract feature importances
feature_names = xgb_model._feature_names
importances = xgb_model._model.feature_importances_

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Calculate percentile threshold (bottom 20%)
threshold_20 = importance_df['importance'].quantile(0.20)
importance_df['keep'] = importance_df['importance'] >= threshold_20

print(f"Total features: {len(importance_df)}")
print(f"20th percentile threshold: {threshold_20:.6f}")
print(f"Features to keep: {importance_df['keep'].sum()}")
print(f"Features to remove: {(~importance_df['keep']).sum()}")

# Show top 20 features
print("\nTop 20 Features:")
importance_df.head(20)
```

Cell 6 (code):
```python
# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Left: Top 30 features
ax1 = axes[0]
top_30 = importance_df.head(30)
colors = ['steelblue' if k else 'lightcoral' for k in top_30['keep']]
ax1.barh(range(len(top_30)), top_30['importance'], color=colors)
ax1.set_yticks(range(len(top_30)))
ax1.set_yticklabels(top_30['feature'])
ax1.invert_yaxis()
ax1.set_xlabel('Importance')
ax1.set_title('Top 30 Features by Importance', fontweight='bold')
ax1.axvline(x=threshold_20, color='red', linestyle='--', label=f'20th percentile ({threshold_20:.4f})')
ax1.legend()

# Right: Bottom features (to be removed)
ax2 = axes[1]
bottom = importance_df[~importance_df['keep']].tail(20)
ax2.barh(range(len(bottom)), bottom['importance'], color='lightcoral')
ax2.set_yticks(range(len(bottom)))
ax2.set_yticklabels(bottom['feature'])
ax2.invert_yaxis()
ax2.set_xlabel('Importance')
ax2.set_title('Bottom Features (To Remove)', fontweight='bold')

plt.tight_layout()
plt.show()

# List features to remove
print("\nFeatures to REMOVE (bottom 20%):")
for f in importance_df[~importance_df['keep']]['feature'].tolist():
    print(f"  - {f}")
```

---

### Task 4: Add new feature engineering section

**Files:**
- Modify: `notebooks/09_feature_optimization.ipynb`

**Step 1: Add new features markdown and code**

Cell 7 (markdown):
```markdown
## 2. New Feature Engineering

Adding 14 new features in 3 categories:
- **Time-based (4):** week_of_month, days_to_month_end, days_from_month_start, quarter
- **Interaction (4):** weekend_event, payday_weekend, month_end_rolling, high_importance_weekend
- **Advanced lags (6):** lag_2, lag_3, lag_28, lag_diff_7, lag_diff_28, lag_ratio_7
```

Cell 8 (code):
```python
def add_new_features(df: pd.DataFrame, target: str = 'daily_logins') -> pd.DataFrame:
    """Add new engineered features to dataframe."""
    df = df.copy()
    dates = pd.to_datetime(df['date'])

    # === TIME-BASED FEATURES ===
    # Week of month (1-5)
    df['week_of_month'] = ((dates.dt.day - 1) // 7) + 1

    # Days to/from month boundaries
    df['days_to_month_end'] = dates.dt.daysinmonth - dates.dt.day
    df['days_from_month_start'] = dates.dt.day - 1

    # Quarter
    df['quarter'] = dates.dt.quarter

    # === INTERACTION FEATURES ===
    # Weekend + any event
    has_event = (df['event_count'] > 0).astype(int) if 'event_count' in df.columns else 0
    df['weekend_event'] = df['is_weekend'] * has_event

    # Payday proximity + weekend
    df['payday_weekend'] = ((df['days_to_payday'] <= 2) & (df['is_weekend'] == 1)).astype(int)

    # Month end + rolling mean
    df['month_end_rolling'] = df['is_month_end'] * df.get(f'{target}_rolling_mean_7', 0)

    # High importance event + weekend
    high_importance = (df['event_importance'] >= 3).astype(int) if 'event_importance' in df.columns else 0
    df['high_importance_weekend'] = high_importance * df['is_weekend']

    # === ADVANCED LAG FEATURES ===
    # Short-term lags
    df[f'{target}_lag_2'] = df[target].shift(2)
    df[f'{target}_lag_3'] = df[target].shift(3)

    # Monthly lag (4 weeks)
    df[f'{target}_lag_28'] = df[target].shift(28)

    # Lag differences (momentum)
    if f'{target}_lag_7' in df.columns:
        df[f'{target}_lag_diff_7'] = df[target] - df[f'{target}_lag_7']
    else:
        df[f'{target}_lag_diff_7'] = df[target] - df[target].shift(7)

    df[f'{target}_lag_diff_28'] = df[target] - df[f'{target}_lag_28']

    # Lag ratio (relative change)
    lag_7 = df[f'{target}_lag_7'] if f'{target}_lag_7' in df.columns else df[target].shift(7)
    df[f'{target}_lag_ratio_7'] = df[target] / lag_7.replace(0, np.nan)

    return df

# Apply new features
df_enhanced = add_new_features(df_features, TARGET)

# List new features
NEW_FEATURES = [
    'week_of_month', 'days_to_month_end', 'days_from_month_start', 'quarter',
    'weekend_event', 'payday_weekend', 'month_end_rolling', 'high_importance_weekend',
    f'{TARGET}_lag_2', f'{TARGET}_lag_3', f'{TARGET}_lag_28',
    f'{TARGET}_lag_diff_7', f'{TARGET}_lag_diff_28', f'{TARGET}_lag_ratio_7'
]

print(f"Added {len(NEW_FEATURES)} new features:")
for f in NEW_FEATURES:
    print(f"  + {f}")
print(f"\nDataset shape: {df_features.shape} -> {df_enhanced.shape}")
```

---

### Task 5: Add feature selection and optimized feature list

**Files:**
- Modify: `notebooks/09_feature_optimization.ipynb`

**Step 1: Add feature selection section**

Cell 9 (markdown):
```markdown
## 3. Feature Selection

Combine new features with existing high-importance features, removing bottom 20%.
```

Cell 10 (code):
```python
# Get features to keep from baseline analysis
features_to_keep = importance_df[importance_df['keep']]['feature'].tolist()
features_to_remove = importance_df[~importance_df['keep']]['feature'].tolist()

# Build optimized external features list
# Start with features that were kept
optimized_external = [f for f in EXTERNAL_FEATURES if f in features_to_keep or f not in importance_df['feature'].tolist()]

# Add new features (except target lags which are handled internally)
new_external = [f for f in NEW_FEATURES if not f.startswith(f'{TARGET}_lag')]
optimized_external.extend(new_external)

# New lag features for model
NEW_LAGS = [2, 3, 28]  # Additional lags beyond [1, 7, 14]
OPTIMIZED_LAGS = [1, 2, 3, 7, 14, 28]

print(f"Original external features: {len(EXTERNAL_FEATURES)}")
print(f"Features removed (low importance): {len(features_to_remove)}")
print(f"New features added: {len(new_external)}")
print(f"Optimized external features: {len(optimized_external)}")
print(f"\nOptimized lags: {OPTIMIZED_LAGS}")

# Save optimized feature list
optimized_features_path = project_root / "data" / "processed" / "selected_features.json"
with open(optimized_features_path, 'w') as f:
    json.dump({
        'external_features': optimized_external,
        'lags': OPTIMIZED_LAGS,
        'removed_features': features_to_remove,
        'new_features': NEW_FEATURES
    }, f, indent=2)
print(f"\nSaved to: {optimized_features_path}")
```

---

### Task 6: Add re-analysis with new features

**Files:**
- Modify: `notebooks/09_feature_optimization.ipynb`

**Step 1: Add importance re-analysis cell**

Cell 11 (code):
```python
# Train XGBoost with ALL features (original + new) to re-analyze importance
ALL_EXTERNAL = EXTERNAL_FEATURES + new_external

xgb_full = XGBoostModel(
    n_estimators=88,
    max_depth=3,
    learning_rate=0.1105,
    min_child_weight=8,
    lags=OPTIMIZED_LAGS,
    external_features=ALL_EXTERNAL,
    name='XGBoost_Full'
)

# Train on first 365 days
train_enhanced = df_enhanced.iloc[:365].copy()
xgb_full.fit(train_enhanced, TARGET)

# Get new importance ranking
full_feature_names = xgb_full._feature_names
full_importances = xgb_full._model.feature_importances_

full_importance_df = pd.DataFrame({
    'feature': full_feature_names,
    'importance': full_importances
}).sort_values('importance', ascending=False)

# Mark new features
full_importance_df['is_new'] = full_importance_df['feature'].apply(
    lambda x: x in NEW_FEATURES or any(x.endswith(f'_lag_{l}') for l in NEW_LAGS)
)

print("Feature Importance with New Features:")
print(f"Total features: {len(full_importance_df)}")
print(f"\nNew features in top 20:")
top_20_new = full_importance_df.head(20)[full_importance_df.head(20)['is_new']]
for _, row in top_20_new.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")
```

Cell 12 (code):
```python
# Visualize new importance ranking
fig, ax = plt.subplots(figsize=(12, 12))

top_40 = full_importance_df.head(40)
colors = ['green' if is_new else 'steelblue' for is_new in top_40['is_new']]

ax.barh(range(len(top_40)), top_40['importance'], color=colors)
ax.set_yticks(range(len(top_40)))
ax.set_yticklabels(top_40['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Top 40 Features (Green = New Features)', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', label='Original'),
    Patch(facecolor='green', label='New')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.show()

# Save full importance
importance_path = project_root / "data" / "processed" / "feature_importance.csv"
full_importance_df.to_csv(importance_path, index=False)
print(f"Saved importance ranking to: {importance_path}")
```

---

### Task 7: Add model comparison with optimized features

**Files:**
- Modify: `notebooks/09_feature_optimization.ipynb`

**Step 1: Add final model comparison section**

Cell 13 (markdown):
```markdown
## 4. Model Comparison with Optimized Features

Train ensemble with optimized feature set and compare to baseline 9.44% MAPE.
```

Cell 14 (code):
```python
# Final optimized feature set (apply 20% threshold to full set)
threshold_full = full_importance_df['importance'].quantile(0.20)
final_features_df = full_importance_df[full_importance_df['importance'] >= threshold_full]
FINAL_EXTERNAL = [f for f in final_features_df['feature'].tolist()
                  if not f.startswith(f'{TARGET}_lag_')]

# Remove lag features from external (they're handled by lags parameter)
FINAL_EXTERNAL = [f for f in FINAL_EXTERNAL if '_lag_' not in f or 'diff' in f or 'ratio' in f]

print(f"Final external features: {len(FINAL_EXTERNAL)}")
print(f"Final lags: {OPTIMIZED_LAGS}")
```

Cell 15 (code):
```python
# Prophet regressors (subset of external features)
PROPHET_REGRESSORS = [f for f in FINAL_EXTERNAL if f in [
    'is_bank_holiday', 'is_racing_event', 'is_tennis_event',
    'is_boxing_event', 'is_football_match', 'event_importance',
    'any_event_tomorrow', 'any_event_in_2_days', 'any_event_in_3_days',
    'bank_holiday_tomorrow', 'bank_holiday_in_2_days', 'bank_holiday_in_3_days',
    'football_tomorrow', 'football_in_2_days', 'football_in_3_days',
    'any_event_yesterday', 'any_event_2_days_ago',
    'bank_holiday_yesterday', 'bank_holiday_2_days_ago',
    'football_yesterday', 'football_2_days_ago',
    'weekend_event', 'high_importance_weekend', 'payday_weekend'
]]

print(f"Prophet regressors: {len(PROPHET_REGRESSORS)}")
```

Cell 16 (code):
```python
# Define models with optimized features
models = []

# XGBoost Optimized (tuned params + optimized features)
models.append(XGBoostModel(
    n_estimators=88, max_depth=3, learning_rate=0.1105, min_child_weight=8,
    lags=OPTIMIZED_LAGS,
    external_features=FINAL_EXTERNAL,
    name='XGBoost_Optimized'
))

# LightGBM Optimized
models.append(LightGBMModel(
    n_estimators=299, max_depth=5, learning_rate=0.0115, num_leaves=52,
    lags=OPTIMIZED_LAGS,
    external_features=FINAL_EXTERNAL,
    name='LightGBM_Optimized'
))

# Prophet Optimized
models.append(ProphetModel(
    yearly_seasonality=True, weekly_seasonality=True,
    regressors=PROPHET_REGRESSORS,
    seasonality_prior_scale=0.0101, seasonality_mode='additive',
    changepoint_prior_scale=0.0117, n_changepoints=44, changepoint_range=0.797,
    name='Prophet_Optimized'
))

# Ensemble Optimized
ensemble_models = [
    XGBoostModel(
        n_estimators=88, max_depth=3, learning_rate=0.1105, min_child_weight=8,
        lags=OPTIMIZED_LAGS, external_features=FINAL_EXTERNAL, name='Ens_XGB_Opt'
    ),
    LightGBMModel(
        n_estimators=299, max_depth=5, learning_rate=0.0115, num_leaves=52,
        lags=OPTIMIZED_LAGS, external_features=FINAL_EXTERNAL, name='Ens_LGB_Opt'
    ),
    ProphetModel(
        yearly_seasonality=True, weekly_seasonality=True, regressors=PROPHET_REGRESSORS,
        seasonality_prior_scale=0.0101, seasonality_mode='additive',
        changepoint_prior_scale=0.0117, n_changepoints=44, changepoint_range=0.797,
        name='Ens_Prophet_Opt'
    )
]
models.append(EnsembleModel(models=ensemble_models, weighting='inverse_mae', name='Ensemble_Optimized'))

print(f"Models to evaluate: {len(models)}")
for m in models:
    print(f"  - {m.name}")
```

---

### Task 8: Add walk-forward validation and results

**Files:**
- Modify: `notebooks/09_feature_optimization.ipynb`

**Step 1: Add validation cell**

Cell 17 (code):
```python
# Walk-forward validation
validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)
benchmark = ModelBenchmark(models=models, validator=validator)

print(f"Running benchmark with {len(models)} models...")
print("This may take a few minutes...\n")

results_df = benchmark.benchmark(
    df_enhanced,
    target=TARGET,
    date_column='date',
    feature_columns=FINAL_EXTERNAL + ['date']
)

print("Benchmark complete!")
```

Cell 18 (code):
```python
# Display results
results_df = results_df.sort_values('mape_mean')

print("\n" + "=" * 70)
print("OPTIMIZED MODEL RESULTS")
print("=" * 70)
print(f"\nBaseline (Ensemble): 9.44% MAPE")
print(f"Target: < 9.00% MAPE\n")

for _, row in results_df.iterrows():
    improvement = 9.44 - row['mape_mean']
    status = "IMPROVED" if row['mape_mean'] < 9.44 else "no change"
    target_hit = "TARGET HIT!" if row['mape_mean'] < 9.0 else ""
    print(f"{row['model_name']:25s} MAPE: {row['mape_mean']:.2f}% (+/- {row['mape_std']:.2f}) [{status}] {target_hit}")
```

---

### Task 9: Add final summary and save results

**Files:**
- Modify: `notebooks/09_feature_optimization.ipynb`

**Step 1: Add summary cells**

Cell 19 (markdown):
```markdown
## 5. Summary and Conclusions
```

Cell 20 (code):
```python
# Final summary
best_model = results_df.iloc[0]
best_mape = best_model['mape_mean']
baseline_mape = 9.44

print("=" * 70)
print("FEATURE OPTIMIZATION SUMMARY")
print("=" * 70)

print(f"""
BASELINE:
  - Best model: Ensemble
  - MAPE: {baseline_mape:.2f}%

OPTIMIZED:
  - Best model: {best_model['model_name']}
  - MAPE: {best_mape:.2f}%

IMPROVEMENT: {baseline_mape - best_mape:.2f} percentage points
RELATIVE IMPROVEMENT: {((baseline_mape - best_mape) / baseline_mape) * 100:.1f}%

TARGET (<9.0%): {"ACHIEVED!" if best_mape < 9.0 else "NOT ACHIEVED"}

KEY CHANGES:
  - Added {len(NEW_FEATURES)} new features
  - Optimized lags: {OPTIMIZED_LAGS}
  - Removed low-importance features
  - Final feature count: {len(FINAL_EXTERNAL)} external + {len(OPTIMIZED_LAGS)} lags
""")

# Save results
results_path = project_root / "data" / "processed" / "optimized_model_results.csv"
results_df.to_csv(results_path, index=False)
print(f"Results saved to: {results_path}")
```

Cell 21 (code):
```python
# Visualization: Before vs After
fig, ax = plt.subplots(figsize=(10, 6))

# Baseline results
baseline_models = ['Ensemble', 'XGBoost_Tuned', 'LightGBM_Tuned', 'Prophet_Tuned']
baseline_mapes = [9.44, 9.48, 10.36, 10.61]

# Optimized results
opt_models = results_df['model_name'].tolist()
opt_mapes = results_df['mape_mean'].tolist()

x = np.arange(len(baseline_models))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_mapes, width, label='Baseline', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, opt_mapes[:len(baseline_models)], width, label='Optimized', color='green', alpha=0.8)

ax.axhline(y=9.0, color='red', linestyle='--', linewidth=2, label='Target (9.0%)')
ax.axhline(y=9.44, color='orange', linestyle='--', linewidth=1, label='Baseline Best (9.44%)')

ax.set_ylabel('MAPE (%)')
ax.set_xlabel('Model')
ax.set_title('Feature Optimization: Before vs After', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Ensemble', 'XGBoost', 'LightGBM', 'Prophet'])
ax.legend()
ax.set_ylim(8, 12)

plt.tight_layout()
plt.show()
```

---

### Task 10: Run notebook and verify results

**Step 1: Execute the notebook**

Run: `jupyter nbconvert --to notebook --execute notebooks/09_feature_optimization.ipynb --ExecutePreprocessor.timeout=600 --inplace`

**Step 2: Verify results**

Check that:
- `data/processed/feature_importance.csv` exists
- `data/processed/selected_features.json` exists
- `data/processed/optimized_model_results.csv` exists
- Final MAPE is reported

---

### Task 11: Commit and push changes

**Step 1: Stage and commit**

```bash
git add notebooks/09_feature_optimization.ipynb
git add data/processed/feature_importance.csv
git add data/processed/selected_features.json
git add data/processed/optimized_model_results.csv
git add docs/plans/2026-01-06-feature-optimization-design.md
git add docs/plans/2026-01-06-feature-optimization-implementation.md
git commit -m "feat: add feature optimization notebook with selection and engineering

- Analyze feature importance using XGBoost
- Add 14 new features (time-based, interaction, advanced lags)
- Remove bottom 20% low-importance features
- Compare optimized models against 9.44% baseline

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

**Step 2: Push to remote**

```bash
git push
```
