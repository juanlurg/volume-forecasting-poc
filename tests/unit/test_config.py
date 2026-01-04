"""Tests for configuration module."""

import pytest


class TestConstants:
    """Test suite for constants."""

    def test_day_multipliers_sum_to_seven(self) -> None:
        """Day multipliers should average to ~1.0."""
        from volume_forecast.config.constants import DAY_OF_WEEK_MULTIPLIERS

        avg = sum(DAY_OF_WEEK_MULTIPLIERS.values()) / 7
        assert 0.9 <= avg <= 1.1

    def test_event_importance_levels_ordered(self) -> None:
        """Event importance levels should be ordered low to major."""
        from volume_forecast.config.constants import EVENT_IMPORTANCE

        assert EVENT_IMPORTANCE["low"] < EVENT_IMPORTANCE["medium"]
        assert EVENT_IMPORTANCE["medium"] < EVENT_IMPORTANCE["high"]
        assert EVENT_IMPORTANCE["high"] < EVENT_IMPORTANCE["major"]

    def test_customer_segments_sum_to_100(self) -> None:
        """Customer segment percentages should sum to 100."""
        from volume_forecast.config.constants import CUSTOMER_SEGMENTS

        total = sum(seg["percentage"] for seg in CUSTOMER_SEGMENTS.values())
        assert total == 100


class TestSettings:
    """Test suite for settings."""

    def test_settings_loads_defaults(self) -> None:
        """Settings should load with sensible defaults."""
        from volume_forecast.config.settings import Settings

        settings = Settings()
        assert settings.forecast_horizon == 7
        assert settings.random_seed == 42

    def test_settings_paths_are_pathlib(self) -> None:
        """Settings paths should be Path objects."""
        from pathlib import Path

        from volume_forecast.config.settings import Settings

        settings = Settings()
        assert isinstance(settings.data_dir, Path)
