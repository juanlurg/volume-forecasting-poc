"""Base generator class for synthetic data."""

from datetime import date, timedelta
from typing import Any

import numpy as np


class BaseGenerator:
    """Base class for data generators with reproducible random state."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_seed(self) -> None:
        """Reset the random number generator to initial seed."""
        self.rng = np.random.default_rng(self.seed)

    def generate_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """Generate a list of dates in a range.

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            List of dates.
        """
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def add_noise(
        self,
        values: np.ndarray,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """Add Gaussian noise to values.

        Args:
            values: Array of values.
            noise_level: Standard deviation as fraction of mean.

        Returns:
            Values with added noise.
        """
        noise = self.rng.normal(0, noise_level * np.mean(values), len(values))
        return values + noise

    def get_metadata(self) -> dict[str, Any]:
        """Get generator metadata for reproducibility.

        Returns:
            Dictionary of generator parameters.
        """
        return {
            "seed": self.seed,
            "generator_class": self.__class__.__name__,
        }
