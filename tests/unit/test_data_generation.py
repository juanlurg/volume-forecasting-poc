"""Tests for data generation module."""

from datetime import date

import numpy as np
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
