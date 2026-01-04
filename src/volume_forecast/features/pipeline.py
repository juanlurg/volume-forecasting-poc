"""Feature engineering pipeline."""

from typing import Any, Self

import pandas as pd

from volume_forecast.features.base import BaseTransformer
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
    ) -> None:
        """Initialize the feature pipeline.

        Args:
            date_column: Name of date column.
            target_columns: Columns to create lag/rolling features for.
            lags: Lag periods to create.
            rolling_windows: Rolling window sizes.
            rolling_stats: Rolling statistics to compute.
            cyclical: Whether to add cyclical temporal features.
        """
        super().__init__()
        self.date_column = date_column
        self.target_columns = target_columns or ["daily_logins", "daily_deposits"]
        self.lags = lags or [1, 7, 14, 21]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.rolling_stats = rolling_stats or ["mean", "std"]
        self.cyclical = cyclical

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

        # Collect feature names
        self._all_feature_names = (
            self._temporal.get_feature_names()
            + self._lags.get_feature_names()
            + self._rolling.get_feature_names()
        )

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
        }
