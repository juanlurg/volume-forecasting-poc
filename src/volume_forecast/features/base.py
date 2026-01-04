"""Base transformer class for feature engineering."""

from abc import ABC, abstractmethod
from typing import Any, Self

import pandas as pd


class BaseTransformer(ABC):
    """Abstract base class for feature transformers.

    Follows scikit-learn transformer interface for compatibility.
    """

    def __init__(self) -> None:
        """Initialize the transformer."""
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the transformer to data.

        Args:
            df: Input DataFrame.
            y: Target variable (optional, for compatibility).

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        pass

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Input DataFrame.
            y: Target variable (optional).

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df, y).transform(df)

    def get_feature_names(self) -> list[str]:
        """Get names of features created by this transformer.

        Returns:
            List of feature names.
        """
        return []

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters.

        Returns:
            Dictionary of parameters.
        """
        return {}
