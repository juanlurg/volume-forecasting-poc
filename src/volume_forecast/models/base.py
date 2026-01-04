"""Base model class for forecasting models."""

from abc import ABC, abstractmethod
from typing import Any, Self

import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all forecasting models.

    All forecasting models should inherit from this class and implement
    the required abstract methods: fit() and predict().

    Attributes:
        _name: The name of the model.
    """

    def __init__(self, name: str) -> None:
        """Initialize the model.

        Args:
            name: The name of the model.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Return the model name.

        Returns:
            The model name.
        """
        return self._name

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target: str) -> Self:
        """Fit the model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Generate predictions.

        Args:
            df: DataFrame containing features for prediction.
            horizon: Number of periods to forecast.

        Returns:
            DataFrame with predictions.
        """
        pass

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """
        return {"name": self._name}
