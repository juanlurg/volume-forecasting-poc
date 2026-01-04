"""Tests for data generation module."""

from datetime import date

import numpy as np
import pandas as pd
import pytest


class TestBaseGenerator:
    """Test suite for base generator."""

    def test_reproducible_with_seed(self) -> None:
        """Generator should produce reproducible results with same seed."""
        from volume_forecast.data_generation.base import BaseGenerator

        gen1 = BaseGenerator(seed=42)
        gen2 = BaseGenerator(seed=42)

        result1 = gen1.rng.random(10)
        result2 = gen2.rng.random(10)

        np.testing.assert_array_equal(result1, result2)

    def test_date_range_generation(self) -> None:
        """Should generate correct date range."""
        from volume_forecast.data_generation.base import BaseGenerator

        gen = BaseGenerator(seed=42)
        dates = gen.generate_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
        )

        assert len(dates) == 10
        assert dates[0] == date(2024, 1, 1)
        assert dates[-1] == date(2024, 1, 10)


class TestVolumeGenerator:
    """Test suite for volume generator."""

    def test_generates_correct_columns(self) -> None:
        """Should generate DataFrame with required columns."""
        from volume_forecast.data_generation.generator import VolumeGenerator

        gen = VolumeGenerator(seed=42)
        df = gen.generate(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert "date" in df.columns
        assert "daily_logins" in df.columns
        assert "daily_deposits" in df.columns
        assert "daily_deposit_volume_gbp" in df.columns

    def test_generates_correct_length(self) -> None:
        """Should generate correct number of rows."""
        from volume_forecast.data_generation.generator import VolumeGenerator

        gen = VolumeGenerator(seed=42)
        df = gen.generate(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert len(df) == 31

    def test_weekend_has_higher_volume(self) -> None:
        """Weekend should have higher average volume than weekdays."""
        from volume_forecast.data_generation.generator import VolumeGenerator

        gen = VolumeGenerator(seed=42)
        df = gen.generate(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        weekend_avg = df[df["day_of_week"] >= 5]["daily_logins"].mean()
        weekday_avg = df[df["day_of_week"] < 5]["daily_logins"].mean()

        assert weekend_avg > weekday_avg

    def test_values_are_positive(self) -> None:
        """All generated values should be positive."""
        from volume_forecast.data_generation.generator import VolumeGenerator

        gen = VolumeGenerator(seed=42)
        df = gen.generate(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert (df["daily_logins"] > 0).all()
        assert (df["daily_deposits"] > 0).all()
        assert (df["daily_deposit_volume_gbp"] > 0).all()
