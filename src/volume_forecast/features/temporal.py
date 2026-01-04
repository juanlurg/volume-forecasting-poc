"""Temporal feature engineering."""

from typing import Any

import numpy as np
import pandas as pd

from volume_forecast.features.base import BaseTransformer


class TemporalFeatures(BaseTransformer):
    """Extract temporal features from date column."""

    def __init__(
        self,
        date_column: str = "date",
        cyclical: bool = True,
        include_payday: bool = True,
    ) -> None:
        """Initialize temporal features transformer.

        Args:
            date_column: Name of date column.
            cyclical: Whether to add cyclical sin/cos features.
            include_payday: Whether to add payday proximity feature.
        """
        super().__init__()
        self.date_column = date_column
        self.cyclical = cyclical
        self.include_payday = include_payday
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame by adding temporal features.

        Args:
            df: Input DataFrame with date column.

        Returns:
            DataFrame with added temporal features.
        """
        df = df.copy()
        dates = pd.to_datetime(df[self.date_column])

        # Basic temporal features
        df["day_of_week"] = dates.dt.dayofweek
        df["day_of_month"] = dates.dt.day
        df["month"] = dates.dt.month
        df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        df["is_month_start"] = dates.dt.is_month_start.astype(int)
        df["is_month_end"] = dates.dt.is_month_end.astype(int)

        self._feature_names = [
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "is_weekend",
            "is_month_start",
            "is_month_end",
        ]

        # Payday proximity (15th and last day of month)
        if self.include_payday:
            days_to_15 = 15 - df["day_of_month"]
            days_to_end = dates.dt.daysinmonth - df["day_of_month"]
            df["days_to_payday"] = np.minimum(
                np.abs(days_to_15), np.abs(days_to_end)
            )
            self._feature_names.append("days_to_payday")

        # Cyclical features
        if self.cyclical:
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
            df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

            self._feature_names.extend([
                "day_of_week_sin",
                "day_of_week_cos",
                "month_sin",
                "month_cos",
                "week_sin",
                "week_cos",
            ])

        return df

    def get_feature_names(self) -> list[str]:
        """Get names of features created."""
        return self._feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters."""
        return {
            "date_column": self.date_column,
            "cyclical": self.cyclical,
            "include_payday": self.include_payday,
        }
