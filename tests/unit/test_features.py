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
