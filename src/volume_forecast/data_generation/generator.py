"""Main volume data generator."""

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volume_forecast.config.constants import (
    BASE_DAILY_DEPOSITS,
    BASE_DAILY_DEPOSIT_VOLUME_GBP,
    BASE_DAILY_LOGINS,
    COL_DATE,
    COL_DEPOSIT_VOLUME,
    COL_DEPOSITS,
    COL_LOGINS,
    DAY_OF_WEEK_MULTIPLIERS,
    EVENT_VOLUME_MULTIPLIERS,
    MONTHLY_MULTIPLIERS,
)
from volume_forecast.data_generation.base import BaseGenerator
from volume_forecast.external_data.aggregator import EventAggregator


class VolumeGenerator(BaseGenerator):
    """Generator for synthetic login and deposit volumes."""

    def __init__(
        self,
        seed: int = 42,
        base_logins: int = BASE_DAILY_LOGINS,
        base_deposits: int = BASE_DAILY_DEPOSITS,
        base_deposit_volume: int = BASE_DAILY_DEPOSIT_VOLUME_GBP,
        noise_level: float = 0.1,
    ) -> None:
        """Initialize the volume generator.

        Args:
            seed: Random seed.
            base_logins: Base daily login count.
            base_deposits: Base daily deposit count.
            base_deposit_volume: Base daily deposit volume in GBP.
            noise_level: Noise level as fraction of base values.
        """
        super().__init__(seed=seed)
        self.base_logins = base_logins
        self.base_deposits = base_deposits
        self.base_deposit_volume = base_deposit_volume
        self.noise_level = noise_level
        self._event_aggregator: EventAggregator | None = None

    @property
    def event_aggregator(self) -> EventAggregator:
        """Lazy-loaded event aggregator."""
        if self._event_aggregator is None:
            self._event_aggregator = EventAggregator()
        return self._event_aggregator

    def _get_day_multiplier(self, d: date) -> float:
        """Get day-of-week volume multiplier."""
        return DAY_OF_WEEK_MULTIPLIERS.get(d.weekday(), 1.0)

    def _get_month_multiplier(self, d: date) -> float:
        """Get monthly seasonality multiplier."""
        return MONTHLY_MULTIPLIERS.get(d.month, 1.0)

    def _get_event_multiplier(self, d: date, events_cache: dict) -> float:
        """Get event-based volume multiplier."""
        if d not in events_cache:
            return 1.0

        max_multiplier = 1.0
        for event in events_cache[d]:
            importance = event.get("importance", "low")
            multiplier = EVENT_VOLUME_MULTIPLIERS.get(importance, 1.0)
            max_multiplier = max(max_multiplier, multiplier)
        return max_multiplier

    def _build_events_cache(
        self, start_date: date, end_date: date
    ) -> dict[date, list[dict]]:
        """Pre-fetch and cache events for date range."""
        cache: dict[date, list[dict]] = {}
        try:
            events = self.event_aggregator.get_events(
                start_date, end_date, include_football=False
            )
            for event in events:
                event_date = event["date"]
                if event_date not in cache:
                    cache[event_date] = []
                cache[event_date].append(event)
        except Exception:
            pass  # Return empty cache on failure
        return cache

    def generate(
        self,
        start_date: date,
        end_date: date,
        include_events: bool = True,
    ) -> pd.DataFrame:
        """Generate synthetic volume data.

        Args:
            start_date: Start date for generation.
            end_date: End date for generation.
            include_events: Whether to include event effects.

        Returns:
            DataFrame with daily volumes.
        """
        dates = self.generate_date_range(start_date, end_date)
        n_days = len(dates)

        # Pre-cache events
        events_cache = {}
        if include_events:
            events_cache = self._build_events_cache(start_date, end_date)

        # Calculate multipliers for each day
        day_multipliers = np.array([self._get_day_multiplier(d) for d in dates])
        month_multipliers = np.array([self._get_month_multiplier(d) for d in dates])
        event_multipliers = np.array([
            self._get_event_multiplier(d, events_cache) for d in dates
        ])

        # Combined multiplier
        combined = day_multipliers * month_multipliers * event_multipliers

        # Generate base volumes with combined multipliers
        logins = self.base_logins * combined
        deposits = self.base_deposits * combined
        deposit_volume = self.base_deposit_volume * combined

        # Add correlated noise
        noise_logins = self.rng.normal(0, self.noise_level * self.base_logins, n_days)
        noise_deposits = self.rng.normal(0, self.noise_level * self.base_deposits, n_days)
        noise_volume = self.rng.normal(
            0, self.noise_level * self.base_deposit_volume, n_days
        )

        logins = np.maximum(logins + noise_logins, 100)  # Minimum floor
        deposits = np.maximum(deposits + noise_deposits, 10)
        deposit_volume = np.maximum(deposit_volume + noise_volume, 1000)

        # Apply login-deposit correlation (~70% correlation)
        deposit_adjustment = 0.7 * (logins / self.base_logins - 1)
        deposits = deposits * (1 + deposit_adjustment * 0.3)

        # Create DataFrame
        df = pd.DataFrame({
            COL_DATE: dates,
            COL_LOGINS: logins.astype(int),
            COL_DEPOSITS: deposits.astype(int),
            COL_DEPOSIT_VOLUME: deposit_volume.astype(int),
        })

        return df

    def save(
        self,
        df: pd.DataFrame,
        output_path: Path,
        include_metadata: bool = True,
    ) -> None:
        """Save generated data to CSV with optional metadata.

        Args:
            df: DataFrame to save.
            output_path: Output file path.
            include_metadata: Whether to save metadata JSON.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save CSV
        df.to_csv(output_path, index=False)

        # Save metadata
        if include_metadata:
            import json

            metadata = self.get_metadata()
            metadata.update({
                "start_date": df[COL_DATE].min().isoformat()
                if hasattr(df[COL_DATE].min(), "isoformat")
                else str(df[COL_DATE].min()),
                "end_date": df[COL_DATE].max().isoformat()
                if hasattr(df[COL_DATE].max(), "isoformat")
                else str(df[COL_DATE].max()),
                "n_rows": len(df),
                "base_logins": self.base_logins,
                "base_deposits": self.base_deposits,
                "noise_level": self.noise_level,
            })
            metadata_path = output_path.with_suffix(".meta.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

    def get_metadata(self) -> dict[str, Any]:
        """Get generator metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "base_logins": self.base_logins,
            "base_deposits": self.base_deposits,
            "base_deposit_volume": self.base_deposit_volume,
            "noise_level": self.noise_level,
        })
        return metadata
