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
