# Event Features Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate external events (holidays, sports) into the forecasting pipeline and update models to use the full feature set.

**Architecture:** Create an `EventFeatures` transformer that uses `EventAggregator` to generate event columns. Update tree models to accept external features alongside auto-generated lags. Update Prophet to use external regressors. Modify walk-forward validation to pass feature data to models.

**Tech Stack:** Python, pandas, Prophet, XGBoost, LightGBM, python-dotenv

---

## Task 1: Create EventFeatures Transformer

**Files:**
- Create: `src/volume_forecast/features/events.py`
- Test: `tests/unit/test_features.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_features.py`:

```python
class TestEventFeatures:
    """Test suite for event features."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with dates spanning known events."""
        # Include dates around Christmas 2023 and a weekend
        return pd.DataFrame({
            "date": pd.date_range("2023-12-23", periods=10, freq="D"),
            "value": range(10),
        })

    def test_adds_is_bank_holiday(self, sample_df: pd.DataFrame) -> None:
        """Should add is_bank_holiday feature."""
        from volume_forecast.features.events import EventFeatures

        transformer = EventFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        assert "is_bank_holiday" in result.columns
        # Dec 25 (Christmas) should be a holiday
        christmas_idx = 2  # 2023-12-25
        assert result["is_bank_holiday"].iloc[christmas_idx] == 1

    def test_adds_event_type_flags(self, sample_df: pd.DataFrame) -> None:
        """Should add event type flag columns."""
        from volume_forecast.features.events import EventFeatures

        transformer = EventFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        expected_cols = [
            "is_bank_holiday",
            "is_racing_event",
            "is_tennis_event",
            "is_boxing_event",
            "is_football_match",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_adds_event_importance(self, sample_df: pd.DataFrame) -> None:
        """Should add event_importance numeric column."""
        from volume_forecast.features.events import EventFeatures

        transformer = EventFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        assert "event_importance" in result.columns
        # Christmas should have major importance (4)
        christmas_idx = 2
        assert result["event_importance"].iloc[christmas_idx] == 4

    def test_adds_event_count(self, sample_df: pd.DataFrame) -> None:
        """Should add event_count column."""
        from volume_forecast.features.events import EventFeatures

        transformer = EventFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        assert "event_count" in result.columns

    def test_get_feature_names(self, sample_df: pd.DataFrame) -> None:
        """Should return list of generated feature names."""
        from volume_forecast.features.events import EventFeatures

        transformer = EventFeatures(date_column="date")
        transformer.fit_transform(sample_df)
        names = transformer.get_feature_names()

        assert len(names) == 7
        assert "is_bank_holiday" in names
        assert "event_importance" in names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features.py::TestEventFeatures -v`
Expected: FAIL with "cannot import name 'EventFeatures'"

**Step 3: Write minimal implementation**

Create `src/volume_forecast/features/events.py`:

```python
"""Event-based features from external data sources."""

import os
from pathlib import Path
from typing import Any, Self

import pandas as pd
from dotenv import load_dotenv

from volume_forecast.features.base import BaseTransformer
from volume_forecast.external_data import EventAggregator


class EventFeatures(BaseTransformer):
    """Transformer that adds event-based features from external data.

    Creates binary flags for each event type plus importance and count columns.

    Attributes:
        date_column: Name of the date column.
        include_football: Whether to include football matches (requires API key).
    """

    IMPORTANCE_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3, "major": 4}

    def __init__(
        self,
        date_column: str = "date",
        include_football: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the event features transformer.

        Args:
            date_column: Name of the date column.
            include_football: Whether to include football match data.
            cache_dir: Directory for caching API responses.
        """
        super().__init__()
        self.date_column = date_column
        self.include_football = include_football
        self.cache_dir = cache_dir or Path("data/external")
        self._feature_names: list[str] = []

        # Load API key from environment
        load_dotenv()
        self._football_api_key = os.getenv("FOOTBALL_API_KEY", "")

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the transformer (no-op for this transformer).

        Args:
            df: Input DataFrame.
            y: Target variable (optional).

        Returns:
            Self.
        """
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event features to the DataFrame.

        Args:
            df: Input DataFrame with date column.

        Returns:
            DataFrame with event feature columns added.
        """
        df = df.copy()

        # Initialize aggregator
        aggregator = EventAggregator(
            cache_dir=self.cache_dir,
            football_api_key=self._football_api_key if self.include_football else "",
        )

        # Get date range
        dates = pd.to_datetime(df[self.date_column])
        start_date = dates.min().date()
        end_date = dates.max().date()

        # Fetch all events in range
        events = aggregator.get_events(
            start_date=start_date,
            end_date=end_date,
            include_football=self.include_football,
        )

        # Build lookup dictionaries
        event_lookup: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            date_str = event["date"].isoformat()
            if date_str not in event_lookup:
                event_lookup[date_str] = []
            event_lookup[date_str].append(event)

        # Initialize feature columns
        df["is_bank_holiday"] = 0
        df["is_racing_event"] = 0
        df["is_tennis_event"] = 0
        df["is_boxing_event"] = 0
        df["is_football_match"] = 0
        df["event_importance"] = 0
        df["event_count"] = 0

        # Populate features
        for idx, row in df.iterrows():
            date_val = pd.to_datetime(row[self.date_column]).date()
            date_str = date_val.isoformat()

            if date_str in event_lookup:
                day_events = event_lookup[date_str]
                df.at[idx, "event_count"] = len(day_events)

                max_importance = 0
                for event in day_events:
                    event_type = event.get("event_type", "")
                    importance = event.get("importance", "low")
                    importance_val = self.IMPORTANCE_MAP.get(importance, 1)

                    if importance_val > max_importance:
                        max_importance = importance_val

                    if event_type == "holiday":
                        df.at[idx, "is_bank_holiday"] = 1
                    elif event_type == "racing":
                        df.at[idx, "is_racing_event"] = 1
                    elif event_type == "tennis":
                        df.at[idx, "is_tennis_event"] = 1
                    elif event_type == "boxing":
                        df.at[idx, "is_boxing_event"] = 1
                    elif event_type == "football":
                        df.at[idx, "is_football_match"] = 1

                df.at[idx, "event_importance"] = max_importance

        self._feature_names = [
            "is_bank_holiday",
            "is_racing_event",
            "is_tennis_event",
            "is_boxing_event",
            "is_football_match",
            "event_importance",
            "event_count",
        ]

        return df

    def get_feature_names(self) -> list[str]:
        """Get names of features created by this transformer.

        Returns:
            List of feature names.
        """
        return self._feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters.

        Returns:
            Dictionary of parameters.
        """
        return {
            "date_column": self.date_column,
            "include_football": self.include_football,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features.py::TestEventFeatures -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/features/events.py tests/unit/test_features.py
git commit -m "feat: add EventFeatures transformer for external events"
```

---

## Task 2: Export EventFeatures from Features Module

**Files:**
- Modify: `src/volume_forecast/features/__init__.py`

**Step 1: Update exports**

Edit `src/volume_forecast/features/__init__.py`:

```python
"""Feature engineering module."""

from volume_forecast.features.base import BaseTransformer
from volume_forecast.features.events import EventFeatures
from volume_forecast.features.lags import LagFeatures
from volume_forecast.features.pipeline import FeaturePipeline
from volume_forecast.features.rolling import RollingFeatures
from volume_forecast.features.temporal import TemporalFeatures

__all__ = [
    "BaseTransformer",
    "EventFeatures",
    "TemporalFeatures",
    "LagFeatures",
    "RollingFeatures",
    "FeaturePipeline",
]
```

**Step 2: Verify import works**

Run: `python -c "from volume_forecast.features import EventFeatures; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/volume_forecast/features/__init__.py
git commit -m "feat: export EventFeatures from features module"
```

---

## Task 3: Add EventFeatures to FeaturePipeline

**Files:**
- Modify: `src/volume_forecast/features/pipeline.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_features.py` in `TestFeaturePipeline` class:

```python
def test_pipeline_includes_events(self, sample_df: pd.DataFrame) -> None:
    """Pipeline should include event features when enabled."""
    from volume_forecast.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(
        date_column="date",
        target_columns=["daily_logins"],
        include_events=True,
    )
    result = pipeline.fit_transform(sample_df)

    assert "is_bank_holiday" in result.columns
    assert "event_importance" in result.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features.py::TestFeaturePipeline::test_pipeline_includes_events -v`
Expected: FAIL with "unexpected keyword argument 'include_events'"

**Step 3: Update FeaturePipeline**

Edit `src/volume_forecast/features/pipeline.py`:

```python
"""Feature engineering pipeline."""

from pathlib import Path
from typing import Any, Self

import pandas as pd

from volume_forecast.features.base import BaseTransformer
from volume_forecast.features.events import EventFeatures
from volume_forecast.features.lags import LagFeatures
from volume_forecast.features.rolling import RollingFeatures
from volume_forecast.features.temporal import TemporalFeatures


class FeaturePipeline(BaseTransformer):
    """Pipeline combining all feature transformers."""

    def __init__(
        self,
        date_column: str = "date",
        target_columns: list[str] | None = None,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        rolling_stats: list[str] | None = None,
        cyclical: bool = True,
        include_events: bool = False,
        include_football: bool = True,
        events_cache_dir: Path | None = None,
    ) -> None:
        """Initialize the feature pipeline.

        Args:
            date_column: Name of date column.
            target_columns: Columns to create lag/rolling features for.
            lags: Lag periods to create.
            rolling_windows: Rolling window sizes.
            rolling_stats: Rolling statistics to compute.
            cyclical: Whether to add cyclical temporal features.
            include_events: Whether to add event features.
            include_football: Whether to include football data in events.
            events_cache_dir: Cache directory for event API responses.
        """
        super().__init__()
        self.date_column = date_column
        self.target_columns = target_columns or ["daily_logins", "daily_deposits"]
        self.lags = lags or [1, 7, 14, 21]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.rolling_stats = rolling_stats or ["mean", "std"]
        self.cyclical = cyclical
        self.include_events = include_events
        self.include_football = include_football
        self.events_cache_dir = events_cache_dir

        # Initialize transformers
        self._temporal = TemporalFeatures(
            date_column=date_column, cyclical=cyclical
        )
        self._lags = LagFeatures(columns=self.target_columns, lags=self.lags)
        self._rolling = RollingFeatures(
            columns=self.target_columns,
            windows=self.rolling_windows,
            stats=self.rolling_stats,
        )
        self._events: EventFeatures | None = None
        if include_events:
            self._events = EventFeatures(
                date_column=date_column,
                include_football=include_football,
                cache_dir=events_cache_dir,
            )
        self._all_feature_names: list[str] = []

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit all transformers.

        Args:
            df: Input DataFrame.
            y: Target variable (optional).

        Returns:
            Self.
        """
        self._temporal.fit(df)
        self._lags.fit(df)
        self._rolling.fit(df)
        if self._events is not None:
            self._events.fit(df)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all transformers.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame with all features.
        """
        df = df.copy()

        # Apply transformers in order
        df = self._temporal.transform(df)
        df = self._lags.transform(df)
        df = self._rolling.transform(df)
        if self._events is not None:
            df = self._events.transform(df)

        # Collect feature names
        self._all_feature_names = (
            self._temporal.get_feature_names()
            + self._lags.get_feature_names()
            + self._rolling.get_feature_names()
        )
        if self._events is not None:
            self._all_feature_names.extend(self._events.get_feature_names())

        return df

    def get_feature_names(self) -> list[str]:
        """Get all generated feature names.

        Returns:
            List of feature names.
        """
        return self._all_feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get pipeline parameters.

        Returns:
            Dictionary of parameters.
        """
        return {
            "date_column": self.date_column,
            "target_columns": self.target_columns,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "rolling_stats": self.rolling_stats,
            "cyclical": self.cyclical,
            "include_events": self.include_events,
            "include_football": self.include_football,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features.py::TestFeaturePipeline -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/features/pipeline.py tests/unit/test_features.py
git commit -m "feat: add event features support to FeaturePipeline"
```

---

## Task 4: Update Tree Models to Accept External Features

**Files:**
- Modify: `src/volume_forecast/models/tree_models.py`
- Test: `tests/unit/test_models.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_models.py`:

```python
class TestTreeModelsWithFeatures:
    """Test tree models with external features."""

    @pytest.fixture
    def sample_df_with_features(self) -> pd.DataFrame:
        """Create sample DataFrame with external features."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "daily_logins": range(1000, 1060),
            "day_of_week": [i % 7 for i in range(60)],
            "is_weekend": [1 if i % 7 >= 5 else 0 for i in range(60)],
            "event_importance": [0] * 50 + [3] * 10,  # Last 10 days have events
        })
        return df

    def test_lightgbm_uses_external_features(
        self, sample_df_with_features: pd.DataFrame
    ) -> None:
        """LightGBM should use external features when provided."""
        pytest.importorskip("lightgbm")
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel(
            external_features=["day_of_week", "is_weekend", "event_importance"],
            lags=[1, 7],
        )
        model.fit(sample_df_with_features, target="daily_logins")

        # Should not raise
        assert model._model is not None

    def test_xgboost_uses_external_features(
        self, sample_df_with_features: pd.DataFrame
    ) -> None:
        """XGBoost should use external features when provided."""
        pytest.importorskip("xgboost")
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel(
            external_features=["day_of_week", "is_weekend", "event_importance"],
            lags=[1, 7],
        )
        model.fit(sample_df_with_features, target="daily_logins")

        assert model._model is not None

    def test_lightgbm_predict_with_future_features(
        self, sample_df_with_features: pd.DataFrame
    ) -> None:
        """LightGBM should accept future_df for prediction."""
        pytest.importorskip("lightgbm")
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel(
            external_features=["day_of_week", "is_weekend", "event_importance"],
            lags=[1, 7],
        )
        model.fit(sample_df_with_features, target="daily_logins")

        # Create future features
        future_df = pd.DataFrame({
            "date": pd.date_range("2024-03-01", periods=7, freq="D"),
            "day_of_week": [4, 5, 6, 0, 1, 2, 3],
            "is_weekend": [0, 1, 1, 0, 0, 0, 0],
            "event_importance": [0, 0, 0, 0, 0, 0, 0],
        })

        predictions = model.predict(horizon=7, future_df=future_df)
        assert len(predictions) == 7
        assert "prediction" in predictions.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_models.py::TestTreeModelsWithFeatures -v`
Expected: FAIL with "unexpected keyword argument 'external_features'"

**Step 3: Update tree_models.py**

Edit `src/volume_forecast/models/tree_models.py` - update both `LightGBMModel` and `XGBoostModel`:

```python
"""Tree-based forecasting models (LightGBM, XGBoost)."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from volume_forecast.models.base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM-based forecasting model with lag and external features.

    This model uses LightGBM for gradient boosting regression with
    automatically generated lag features and optional external features.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth (-1 for unlimited).
        learning_rate: Boosting learning rate.
        lags: List of lag periods for feature creation.
        external_features: List of external feature column names.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        lags: list[int] | None = None,
        external_features: list[str] | None = None,
        name: str = "lightgbm",
    ) -> None:
        """Initialize the LightGBM model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth (-1 for unlimited).
            learning_rate: Boosting learning rate.
            lags: List of lag periods for feature creation (default: [1, 7, 14]).
            external_features: List of external feature columns to use.
            name: Name of the model.
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lags = lags if lags is not None else [1, 7, 14]
        self.external_features = external_features or []
        self._model: Any = None
        self._target: str | None = None
        self._train_data: pd.DataFrame | None = None
        self._last_train_date: pd.Timestamp | None = None
        self._feature_columns: list[str] = []

    def _create_lag_features(
        self, df: pd.DataFrame, target: str
    ) -> pd.DataFrame:
        """Create lag features from target column.

        Args:
            df: DataFrame with target column.
            target: Name of the target column.

        Returns:
            DataFrame with lag features added.
        """
        result = df.copy()
        for lag in self.lags:
            result[f"lag_{lag}"] = result[target].shift(lag)
        return result

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        feature_columns: list[str] | None = None,
    ) -> Self:
        """Fit the LightGBM model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            feature_columns: Optional list of feature columns. If None,
                lag features + external_features will be used.

        Returns:
            Self for method chaining.
        """
        self._target = target
        self._train_data = train_df.copy()
        self._last_train_date = pd.Timestamp(train_df["date"].iloc[-1])

        # Create lag features
        df_with_features = self._create_lag_features(train_df, target)
        lag_columns = [f"lag_{lag}" for lag in self.lags]

        # Determine feature columns
        if feature_columns is None:
            self._feature_columns = lag_columns + list(self.external_features)
        else:
            self._feature_columns = feature_columns

        # Validate external features exist
        for col in self.external_features:
            if col not in df_with_features.columns:
                raise ValueError(f"External feature '{col}' not found in training data")

        # Drop rows with NaN (from lag creation)
        df_clean = df_with_features.dropna(subset=lag_columns)

        X = df_clean[self._feature_columns].values
        y = df_clean[target].values

        # Import at runtime to avoid import errors if library not available
        from lightgbm import LGBMRegressor

        # Initialize and fit LightGBM model
        self._model = LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=-1,
        )
        self._model.fit(X, y)

        return self

    def predict(
        self,
        df: pd.DataFrame | None = None,
        horizon: int = 7,
        future_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate recursive multi-step forecasts.

        Args:
            df: Optional DataFrame for prediction (not used, kept for API compatibility).
            horizon: Number of periods to forecast.
            future_df: DataFrame with external features for future dates.
                Required if external_features were used in training.

        Returns:
            DataFrame with 'date' and 'prediction' columns.

        Raises:
            ValueError: If model hasn't been fit or future_df missing when needed.
        """
        if self._model is None:
            raise ValueError("Model must be fit before calling predict.")

        # Check if we need future features
        if self.external_features and future_df is None:
            raise ValueError(
                "future_df required when model uses external_features. "
                f"Expected columns: {self.external_features}"
            )

        # Get historical values for lag calculation
        historical_values = list(self._train_data[self._target].values)

        # Generate prediction dates starting from day after last training date
        prediction_dates = pd.date_range(
            start=self._last_train_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        predictions = []
        for step in range(horizon):
            # Create lag features from historical + predicted values
            feature_values = []
            for lag in self.lags:
                if lag <= len(historical_values):
                    feature_values.append(historical_values[-lag])
                else:
                    feature_values.append(historical_values[0])

            # Add external features if present
            if self.external_features and future_df is not None:
                for col in self.external_features:
                    feature_values.append(future_df[col].iloc[step])

            # Predict next value
            X_pred = np.array([feature_values])
            pred = self._model.predict(X_pred)[0]
            predictions.append(pred)

            # Add prediction to history for next step
            historical_values.append(pred)

        return pd.DataFrame({
            "date": prediction_dates,
            "prediction": predictions,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """
        params = super().get_params()
        params.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "lags": self.lags,
            "external_features": self.external_features,
        })
        return params


class XGBoostModel(BaseModel):
    """XGBoost-based forecasting model with lag and external features.

    This model uses XGBoost for gradient boosting regression with
    automatically generated lag features and optional external features.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        lags: List of lag periods for feature creation.
        external_features: List of external feature column names.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        lags: list[int] | None = None,
        external_features: list[str] | None = None,
        name: str = "xgboost",
    ) -> None:
        """Initialize the XGBoost model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            lags: List of lag periods for feature creation (default: [1, 7, 14]).
            external_features: List of external feature columns to use.
            name: Name of the model.
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lags = lags if lags is not None else [1, 7, 14]
        self.external_features = external_features or []
        self._model: Any = None
        self._target: str | None = None
        self._train_data: pd.DataFrame | None = None
        self._last_train_date: pd.Timestamp | None = None
        self._feature_columns: list[str] = []

    def _create_lag_features(
        self, df: pd.DataFrame, target: str
    ) -> pd.DataFrame:
        """Create lag features from target column.

        Args:
            df: DataFrame with target column.
            target: Name of the target column.

        Returns:
            DataFrame with lag features added.
        """
        result = df.copy()
        for lag in self.lags:
            result[f"lag_{lag}"] = result[target].shift(lag)
        return result

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        feature_columns: list[str] | None = None,
    ) -> Self:
        """Fit the XGBoost model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            feature_columns: Optional list of feature columns. If None,
                lag features + external_features will be used.

        Returns:
            Self for method chaining.
        """
        self._target = target
        self._train_data = train_df.copy()
        self._last_train_date = pd.Timestamp(train_df["date"].iloc[-1])

        # Create lag features
        df_with_features = self._create_lag_features(train_df, target)
        lag_columns = [f"lag_{lag}" for lag in self.lags]

        # Determine feature columns
        if feature_columns is None:
            self._feature_columns = lag_columns + list(self.external_features)
        else:
            self._feature_columns = feature_columns

        # Validate external features exist
        for col in self.external_features:
            if col not in df_with_features.columns:
                raise ValueError(f"External feature '{col}' not found in training data")

        # Drop rows with NaN (from lag creation)
        df_clean = df_with_features.dropna(subset=lag_columns)

        X = df_clean[self._feature_columns].values
        y = df_clean[target].values

        # Import at runtime to avoid import errors if library not available
        from xgboost import XGBRegressor

        # Initialize and fit XGBoost model
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=0,
        )
        self._model.fit(X, y)

        return self

    def predict(
        self,
        df: pd.DataFrame | None = None,
        horizon: int = 7,
        future_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate recursive multi-step forecasts.

        Args:
            df: Optional DataFrame for prediction (not used, kept for API compatibility).
            horizon: Number of periods to forecast.
            future_df: DataFrame with external features for future dates.
                Required if external_features were used in training.

        Returns:
            DataFrame with 'date' and 'prediction' columns.

        Raises:
            ValueError: If model hasn't been fit or future_df missing when needed.
        """
        if self._model is None:
            raise ValueError("Model must be fit before calling predict.")

        # Check if we need future features
        if self.external_features and future_df is None:
            raise ValueError(
                "future_df required when model uses external_features. "
                f"Expected columns: {self.external_features}"
            )

        # Get historical values for lag calculation
        historical_values = list(self._train_data[self._target].values)

        # Generate prediction dates starting from day after last training date
        prediction_dates = pd.date_range(
            start=self._last_train_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        predictions = []
        for step in range(horizon):
            # Create lag features from historical + predicted values
            feature_values = []
            for lag in self.lags:
                if lag <= len(historical_values):
                    feature_values.append(historical_values[-lag])
                else:
                    feature_values.append(historical_values[0])

            # Add external features if present
            if self.external_features and future_df is not None:
                for col in self.external_features:
                    feature_values.append(future_df[col].iloc[step])

            # Predict next value
            X_pred = np.array([feature_values])
            pred = self._model.predict(X_pred)[0]
            predictions.append(pred)

            # Add prediction to history for next step
            historical_values.append(pred)

        return pd.DataFrame({
            "date": prediction_dates,
            "prediction": predictions,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """
        params = super().get_params()
        params.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "lags": self.lags,
            "external_features": self.external_features,
        })
        return params
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_models.py::TestTreeModelsWithFeatures -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/models/tree_models.py tests/unit/test_models.py
git commit -m "feat: add external features support to tree models"
```

---

## Task 5: Update Prophet to Support Regressors

**Files:**
- Modify: `src/volume_forecast/models/prophet_model.py`
- Test: `tests/unit/test_models.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_models.py`:

```python
class TestProphetWithRegressors:
    """Test Prophet model with external regressors."""

    @pytest.fixture
    def sample_df_with_regressors(self) -> pd.DataFrame:
        """Create sample DataFrame with regressor columns."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "daily_logins": range(1000, 1060),
            "is_bank_holiday": [0] * 55 + [1] * 5,
            "event_importance": [0] * 50 + [3] * 10,
        })

    def test_prophet_accepts_regressors(
        self, sample_df_with_regressors: pd.DataFrame
    ) -> None:
        """Prophet should accept regressor columns."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel(
            regressors=["is_bank_holiday", "event_importance"],
        )
        model.fit(sample_df_with_regressors, target="daily_logins")

        assert model._model is not None

    def test_prophet_predict_with_future_regressors(
        self, sample_df_with_regressors: pd.DataFrame
    ) -> None:
        """Prophet should use future_df for regressor values."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel(
            regressors=["is_bank_holiday", "event_importance"],
        )
        model.fit(sample_df_with_regressors, target="daily_logins")

        future_df = pd.DataFrame({
            "date": pd.date_range("2024-03-01", periods=7, freq="D"),
            "is_bank_holiday": [0, 0, 0, 0, 0, 0, 0],
            "event_importance": [0, 0, 0, 0, 0, 0, 0],
        })

        predictions = model.predict(horizon=7, future_df=future_df)
        assert len(predictions) == 7
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_models.py::TestProphetWithRegressors -v`
Expected: FAIL with "unexpected keyword argument 'regressors'"

**Step 3: Update prophet_model.py**

Edit `src/volume_forecast/models/prophet_model.py`:

```python
"""Prophet forecasting model wrapper.

This module provides a wrapper around Facebook Prophet for time series forecasting.
Prophet is a procedure for forecasting time series data that is robust to
missing data, shifts in the trend, and large outliers.
"""

import logging
from typing import Any, Self

import pandas as pd
from prophet import Prophet

from volume_forecast.models.base import BaseModel


class ProphetModel(BaseModel):
    """Prophet forecasting model using Facebook Prophet.

    This model wraps Facebook Prophet for time series forecasting with
    automatic seasonality detection, trend modeling, and optional external regressors.

    Attributes:
        _yearly_seasonality: Whether to include yearly seasonality.
        _weekly_seasonality: Whether to include weekly seasonality.
        _daily_seasonality: Whether to include daily seasonality.
        _regressors: List of external regressor column names.
        _model: Fitted Prophet model.
        _last_date: The last date from training data.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        regressors: list[str] | None = None,
        name: str = "prophet",
    ) -> None:
        """Initialize the ProphetModel.

        Args:
            yearly_seasonality: Whether to include yearly seasonality.
            weekly_seasonality: Whether to include weekly seasonality.
            daily_seasonality: Whether to include daily seasonality.
            regressors: List of external regressor column names.
            name: The name of the model.
        """
        super().__init__(name)
        self._yearly_seasonality = yearly_seasonality
        self._weekly_seasonality = weekly_seasonality
        self._daily_seasonality = daily_seasonality
        self._regressors = regressors or []
        self._model: Prophet | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        date_column: str = "date",
    ) -> Self:
        """Fit the Prophet model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            date_column: Name of the date column.

        Returns:
            Self for method chaining.
        """
        # Suppress Prophet logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        # Convert to Prophet format (ds, y columns)
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(train_df[date_column]),
            "y": train_df[target].values,
        })

        # Add regressor columns
        for reg in self._regressors:
            if reg not in train_df.columns:
                raise ValueError(f"Regressor '{reg}' not found in training data")
            prophet_df[reg] = train_df[reg].values

        # Create Prophet model
        self._model = Prophet(
            yearly_seasonality=self._yearly_seasonality,
            weekly_seasonality=self._weekly_seasonality,
            daily_seasonality=self._daily_seasonality,
        )

        # Add regressors before fitting
        for reg in self._regressors:
            self._model.add_regressor(reg)

        # Fit the model
        self._model.fit(prophet_df)

        self._last_date = pd.Timestamp(train_df[date_column].iloc[-1])
        return self

    def predict(
        self,
        horizon: int = 7,
        future_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts using the fitted model.

        Args:
            horizon: Number of periods to forecast.
            future_df: DataFrame with regressor values for future dates.
                Required if regressors were used in training.

        Returns:
            DataFrame with 'date' and 'prediction' columns.

        Raises:
            ValueError: If model hasn't been fit or future_df missing when needed.
        """
        if self._model is None or self._last_date is None:
            raise ValueError("Model must be fit before predicting.")

        # Check if we need future regressor values
        if self._regressors and future_df is None:
            raise ValueError(
                "future_df required when model uses regressors. "
                f"Expected columns: {self._regressors}"
            )

        # Create future dataframe starting from day after last training date
        future_dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        prophet_future = pd.DataFrame({"ds": future_dates})

        # Add regressor values for future dates
        if self._regressors and future_df is not None:
            for reg in self._regressors:
                if reg not in future_df.columns:
                    raise ValueError(
                        f"Regressor '{reg}' not found in future_df"
                    )
                prophet_future[reg] = future_df[reg].values[:horizon]

        # Generate forecast
        forecast = self._model.predict(prophet_future)

        return pd.DataFrame({
            "date": future_dates,
            "prediction": forecast["yhat"].values,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters including seasonality settings.
        """
        params = super().get_params()
        params["yearly_seasonality"] = self._yearly_seasonality
        params["weekly_seasonality"] = self._weekly_seasonality
        params["daily_seasonality"] = self._daily_seasonality
        params["regressors"] = self._regressors
        return params
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_models.py::TestProphetWithRegressors -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/models/prophet_model.py tests/unit/test_models.py
git commit -m "feat: add external regressors support to Prophet model"
```

---

## Task 6: Update WalkForwardValidator to Pass Features

**Files:**
- Modify: `src/volume_forecast/evaluation/validation.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_evaluation.py`:

```python
def test_validator_passes_feature_df(self) -> None:
    """Validator should pass feature data to models that need it."""
    from volume_forecast.evaluation.validation import WalkForwardValidator

    # Create mock model that tracks if it received feature data
    class MockModelWithFeatures:
        def __init__(self):
            self.received_future_df = False

        def fit(self, train_df, target):
            return self

        def predict(self, horizon, future_df=None):
            if future_df is not None:
                self.received_future_df = True
            return pd.DataFrame({
                "date": pd.date_range("2024-06-01", periods=horizon),
                "prediction": [100] * horizon,
            })

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=400, freq="D"),
        "daily_logins": range(400),
        "feature_col": range(400),
    })

    validator = WalkForwardValidator(min_train_size=365, test_size=7)
    model = MockModelWithFeatures()

    # Run validation with feature columns specified
    results = validator.validate(
        model, df, target="daily_logins",
        feature_columns=["feature_col"],
    )

    assert model.received_future_df
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluation.py::test_validator_passes_feature_df -v`
Expected: FAIL with "unexpected keyword argument 'feature_columns'"

**Step 3: Update validation.py**

Edit `src/volume_forecast/evaluation/validation.py` - update the `validate` method:

```python
def validate(
    self,
    model: ForecastModel,
    df: pd.DataFrame,
    target: str,
    date_column: str = "date",
    feature_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run walk-forward validation on a model.

    For each fold:
    1. Fit the model on training data
    2. Generate predictions for the test period
    3. Calculate evaluation metrics

    Args:
        model: A forecast model implementing fit() and predict().
        df: DataFrame containing features and target.
        target: Name of the target column.
        date_column: Name of the date column.
        feature_columns: Optional list of feature columns to pass to model.
            If provided, future feature values will be passed to predict().

    Returns:
        List of dictionaries, one per fold, containing:
        - fold_id: Index of the fold (0-based)
        - metrics: Dictionary of evaluation metrics (mae, rmse, mape, smape)
        - predictions: DataFrame with actual values and predictions
    """
    results: list[dict[str, Any]] = []

    for fold_id, (train_df, test_df) in enumerate(self.split(df, date_column=date_column)):
        # Fit model on training data
        model.fit(train_df, target)

        # Prepare future_df if feature columns are specified
        future_df = None
        if feature_columns:
            future_df = test_df[[date_column] + list(feature_columns)].copy()

        # Generate predictions - try with future_df first, fall back to without
        try:
            if future_df is not None:
                predictions_df = model.predict(horizon=self.test_size, future_df=future_df)
            else:
                predictions_df = model.predict(horizon=self.test_size)
        except TypeError:
            # Model doesn't accept future_df parameter
            predictions_df = model.predict(horizon=self.test_size)

        # Get actual values
        y_true = test_df[target].values

        # Get predicted values (handle different column naming conventions)
        if "prediction" in predictions_df.columns:
            y_pred = predictions_df["prediction"].values
        elif "yhat" in predictions_df.columns:
            y_pred = predictions_df["yhat"].values
        elif target in predictions_df.columns:
            y_pred = predictions_df[target].values
        else:
            # Assume first numeric column is predictions
            numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                y_pred = predictions_df[numeric_cols[0]].values
            else:
                raise ValueError(
                    f"Could not find prediction column in {predictions_df.columns.tolist()}"
                )

        # Calculate metrics
        metrics = calculate_all_metrics(y_true, y_pred)

        # Build result dictionary
        result = {
            "fold_id": fold_id,
            "metrics": metrics,
            "predictions": pd.DataFrame(
                {
                    date_column: test_df[date_column].values,
                    "actual": y_true,
                    "predicted": y_pred,
                }
            ),
        }

        results.append(result)

    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_evaluation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/evaluation/validation.py tests/unit/test_evaluation.py
git commit -m "feat: add feature_columns support to WalkForwardValidator"
```

---

## Task 7: Run All Tests

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Commit if any fixes needed**

```bash
git add -A
git commit -m "fix: resolve test issues"
```

---

## Task 8: Update Model Comparison Notebook

**Files:**
- Modify: `notebooks/07_model_comparison.ipynb`

This task involves updating the notebook to:
1. Use FeaturePipeline with `include_events=True`
2. Instantiate tree models with external features
3. Instantiate Prophet with regressors
4. Re-run the benchmark and compare results

**Key changes to make in the notebook:**

1. After loading data, apply feature pipeline:
```python
from volume_forecast.features import FeaturePipeline

pipeline = FeaturePipeline(
    date_column='date',
    target_columns=['daily_logins'],
    include_events=True,
    include_football=True,
)
df_features = pipeline.fit_transform(df)
```

2. Define feature columns for models:
```python
# Features for tree models
EXTERNAL_FEATURES = [
    'day_of_week', 'is_weekend', 'day_of_week_sin', 'day_of_week_cos',
    'month', 'month_sin', 'month_cos',
    'is_bank_holiday', 'is_racing_event', 'is_tennis_event',
    'is_boxing_event', 'is_football_match', 'event_importance',
]

# Regressors for Prophet
PROPHET_REGRESSORS = [
    'is_bank_holiday', 'is_racing_event', 'event_importance',
]
```

3. Update model instantiation:
```python
models.append(XGBoostModel(
    n_estimators=100, max_depth=6, lags=[1, 7, 14],
    external_features=EXTERNAL_FEATURES,
    name='XGBoost_Enhanced'
))

models.append(LightGBMModel(
    n_estimators=100, max_depth=-1, lags=[1, 7, 14],
    external_features=EXTERNAL_FEATURES,
    name='LightGBM_Enhanced'
))

models.append(ProphetModel(
    yearly_seasonality=True, weekly_seasonality=True,
    regressors=PROPHET_REGRESSORS,
    name='Prophet_Enhanced'
))
```

4. Pass feature columns to validator:
```python
results = validator.validate(
    model, df_features, target=TARGET,
    feature_columns=EXTERNAL_FEATURES + PROPHET_REGRESSORS,
)
```

**Commit:**
```bash
git add notebooks/07_model_comparison.ipynb
git commit -m "feat: update model comparison with enhanced features"
```

---

## Task 9: Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run the notebook end-to-end**

Open and execute `notebooks/07_model_comparison.ipynb` to verify:
- Feature pipeline generates event columns
- Models train without errors
- Results show improved ML model performance

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete event features integration"
```

---

## Summary

This plan implements:
1. **EventFeatures transformer** - Generates 7 event columns from external data
2. **FeaturePipeline update** - Optionally includes event features
3. **Tree model updates** - Accept external features for training and prediction
4. **Prophet update** - Accept external regressors
5. **Validator update** - Pass feature data to models during validation
6. **Notebook update** - Re-run benchmarks with enhanced models

Expected outcome: ML models (XGBoost, LightGBM) should improve significantly in the rankings, potentially competing with or surpassing Prophet.
