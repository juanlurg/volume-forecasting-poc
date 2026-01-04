"""Rolling window feature engineering."""

from typing import Any

import pandas as pd

from volume_forecast.features.base import BaseTransformer


class RollingFeatures(BaseTransformer):
    """Create rolling window statistics features."""

    def __init__(
        self,
        columns: list[str],
        windows: list[int] | None = None,
        stats: list[str] | None = None,
    ) -> None:
        """Initialize rolling features transformer.

        Args:
            columns: Columns to create rolling features for.
            windows: List of window sizes (default: [7, 14, 30]).
            stats: Statistics to compute (default: ["mean", "std"]).
        """
        super().__init__()
        self.columns = columns
        self.windows = windows or [7, 14, 30]
        self.stats = stats or ["mean", "std"]
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame by adding rolling features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added rolling features.
        """
        df = df.copy()
        self._feature_names = []

        for col in self.columns:
            if col not in df.columns:
                continue

            for window in self.windows:
                rolling = df[col].rolling(window=window, min_periods=1)

                for stat in self.stats:
                    feature_name = f"{col}_rolling_{stat}_{window}"

                    if stat == "mean":
                        df[feature_name] = rolling.mean()
                    elif stat == "std":
                        df[feature_name] = rolling.std()
                    elif stat == "min":
                        df[feature_name] = rolling.min()
                    elif stat == "max":
                        df[feature_name] = rolling.max()
                    elif stat == "sum":
                        df[feature_name] = rolling.sum()

                    self._feature_names.append(feature_name)

        return df

    def get_feature_names(self) -> list[str]:
        """Get names of features created."""
        return self._feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters."""
        return {
            "columns": self.columns,
            "windows": self.windows,
            "stats": self.stats,
        }
