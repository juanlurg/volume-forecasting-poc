"""Tests for feature engineering module."""

from datetime import date

import pandas as pd
import pytest


class TestBaseTransformer:
    """Test suite for base transformer."""

    def test_fit_returns_self(self) -> None:
        """fit() should return self for chaining."""
        from volume_forecast.features.base import BaseTransformer

        class DummyTransformer(BaseTransformer):
            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

        transformer = DummyTransformer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = transformer.fit(df)

        assert result is transformer

    def test_fit_transform_chains(self) -> None:
        """fit_transform() should work correctly."""
        from volume_forecast.features.base import BaseTransformer

        class AddOneTransformer(BaseTransformer):
            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                return df.assign(b=df["a"] + 1)

        transformer = AddOneTransformer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = transformer.fit_transform(df)

        assert "b" in result.columns
        assert list(result["b"]) == [2, 3, 4]


class TestTemporalFeatures:
    """Test suite for temporal features."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with dates."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "value": range(30),
        })

    def test_adds_day_of_week(self, sample_df: pd.DataFrame) -> None:
        """Should add day_of_week feature."""
        from volume_forecast.features.temporal import TemporalFeatures

        transformer = TemporalFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        assert "day_of_week" in result.columns
        assert result["day_of_week"].iloc[0] == 0  # Monday

    def test_adds_is_weekend(self, sample_df: pd.DataFrame) -> None:
        """Should add is_weekend feature."""
        from volume_forecast.features.temporal import TemporalFeatures

        transformer = TemporalFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        assert "is_weekend" in result.columns
        # Jan 6, 2024 is Saturday
        saturday_idx = 5  # 6th day (0-indexed)
        assert result["is_weekend"].iloc[saturday_idx] == 1

    def test_adds_cyclical_features(self, sample_df: pd.DataFrame) -> None:
        """Should add sin/cos cyclical features."""
        from volume_forecast.features.temporal import TemporalFeatures

        transformer = TemporalFeatures(date_column="date", cyclical=True)
        result = transformer.fit_transform(sample_df)

        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns


class TestLagFeatures:
    """Test suite for lag features."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "daily_logins": range(100, 130),
        })

    def test_creates_lag_columns(self, sample_df: pd.DataFrame) -> None:
        """Should create lag columns."""
        from volume_forecast.features.lags import LagFeatures

        transformer = LagFeatures(columns=["daily_logins"], lags=[1, 7])
        result = transformer.fit_transform(sample_df)

        assert "daily_logins_lag_1" in result.columns
        assert "daily_logins_lag_7" in result.columns

    def test_lag_values_correct(self, sample_df: pd.DataFrame) -> None:
        """Lag values should be shifted correctly."""
        from volume_forecast.features.lags import LagFeatures

        transformer = LagFeatures(columns=["daily_logins"], lags=[1])
        result = transformer.fit_transform(sample_df)

        # Row 1's lag_1 should equal row 0's value
        assert result["daily_logins_lag_1"].iloc[1] == sample_df["daily_logins"].iloc[0]

    def test_handles_missing_at_start(self, sample_df: pd.DataFrame) -> None:
        """Should handle missing values at series start."""
        from volume_forecast.features.lags import LagFeatures

        transformer = LagFeatures(columns=["daily_logins"], lags=[7])
        result = transformer.fit_transform(sample_df)

        # First 7 values should be NaN
        assert result["daily_logins_lag_7"].iloc[:7].isna().all()
        assert result["daily_logins_lag_7"].iloc[7:].notna().all()


class TestRollingFeatures:
    """Test suite for rolling features."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "daily_logins": [100] * 30,  # Constant for easy verification
        })

    def test_creates_rolling_mean(self, sample_df: pd.DataFrame) -> None:
        """Should create rolling mean columns."""
        from volume_forecast.features.rolling import RollingFeatures

        transformer = RollingFeatures(columns=["daily_logins"], windows=[7])
        result = transformer.fit_transform(sample_df)

        assert "daily_logins_rolling_mean_7" in result.columns

    def test_rolling_mean_value(self, sample_df: pd.DataFrame) -> None:
        """Rolling mean should be correct."""
        from volume_forecast.features.rolling import RollingFeatures

        transformer = RollingFeatures(columns=["daily_logins"], windows=[7])
        result = transformer.fit_transform(sample_df)

        # For constant values, rolling mean should equal the value
        assert result["daily_logins_rolling_mean_7"].iloc[7] == 100.0

    def test_creates_rolling_std(self, sample_df: pd.DataFrame) -> None:
        """Should create rolling std columns."""
        from volume_forecast.features.rolling import RollingFeatures

        transformer = RollingFeatures(
            columns=["daily_logins"], windows=[7], stats=["mean", "std"]
        )
        result = transformer.fit_transform(sample_df)

        assert "daily_logins_rolling_std_7" in result.columns


class TestEventFeatures:
    """Test suite for event features."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with dates spanning known events."""
        # Include dates around Christmas 2024 (UK bank holidays available from 2024)
        return pd.DataFrame({
            "date": pd.date_range("2024-12-23", periods=10, freq="D"),
            "value": range(10),
        })

    def test_adds_is_bank_holiday(self, sample_df: pd.DataFrame) -> None:
        """Should add is_bank_holiday feature."""
        from volume_forecast.features.events import EventFeatures

        transformer = EventFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        assert "is_bank_holiday" in result.columns
        # Dec 25 (Christmas 2024) should be a holiday
        christmas_idx = 2  # 2024-12-25
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
        # Christmas 2024 should have major importance (4)
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


class TestFeaturePipeline:
    """Test suite for feature pipeline."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "daily_logins": range(1000, 1060),
            "daily_deposits": range(100, 160),
        })

    def test_pipeline_chains_transformers(self, sample_df: pd.DataFrame) -> None:
        """Pipeline should chain multiple transformers."""
        from volume_forecast.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            date_column="date",
            target_columns=["daily_logins"],
            lags=[1, 7],
            rolling_windows=[7],
        )
        result = pipeline.fit_transform(sample_df)

        # Should have temporal, lag, and rolling features
        assert "day_of_week" in result.columns
        assert "daily_logins_lag_1" in result.columns
        assert "daily_logins_rolling_mean_7" in result.columns

    def test_get_all_feature_names(self, sample_df: pd.DataFrame) -> None:
        """Should return all generated feature names."""
        from volume_forecast.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            date_column="date",
            target_columns=["daily_logins"],
        )
        result = pipeline.fit_transform(sample_df)
        feature_names = pipeline.get_feature_names()

        assert len(feature_names) > 0
        for name in feature_names:
            assert name in result.columns

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
