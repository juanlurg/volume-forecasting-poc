"""Lag feature engineering."""

from typing import Any

import pandas as pd

from volume_forecast.features.base import BaseTransformer


class LagFeatures(BaseTransformer):
    """Create lag features from specified columns."""

    def __init__(
        self,
        columns: list[str],
        lags: list[int] | None = None,
    ) -> None:
        """Initialize lag features transformer.

        Args:
            columns: Columns to create lags for.
            lags: List of lag periods (default: [1, 7, 14, 21]).
        """
        super().__init__()
        self.columns = columns
        self.lags = lags or [1, 7, 14, 21]
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame by adding lag features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added lag features.
        """
        df = df.copy()
        self._feature_names = []

        for col in self.columns:
            if col not in df.columns:
                continue

            for lag in self.lags:
                feature_name = f"{col}_lag_{lag}"
                df[feature_name] = df[col].shift(lag)
                self._feature_names.append(feature_name)

        return df

    def get_feature_names(self) -> list[str]:
        """Get names of features created."""
        return self._feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters."""
        return {
            "columns": self.columns,
            "lags": self.lags,
        }
