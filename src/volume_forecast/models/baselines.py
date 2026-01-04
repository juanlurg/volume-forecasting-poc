"""Baseline forecasting models.

This module provides simple baseline models that serve as benchmarks
for more sophisticated forecasting methods.
"""

from typing import Any, Self

import numpy as np
import pandas as pd

from volume_forecast.models.base import BaseModel


class NaiveModel(BaseModel):
    """Naive forecasting model that predicts the last observed value.

    This model simply repeats the last value from the training data
    for all future predictions. It serves as the simplest baseline.

    Attributes:
        _last_value: The last observed value from training.
        _last_date: The last date from training data.
    """

    def __init__(self, name: str = "naive") -> None:
        """Initialize the NaiveModel.

        Args:
            name: The name of the model.
        """
        super().__init__(name)
        self._last_value: float | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_df: pd.DataFrame, target: str) -> Self:
        """Fit the model by storing the last value.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.

        Returns:
            Self for method chaining.
        """
        self._last_value = train_df[target].iloc[-1]
        self._last_date = pd.Timestamp(train_df["date"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Generate predictions by repeating the last value.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            DataFrame with 'date' and 'prediction' columns.
        """
        if self._last_value is None or self._last_date is None:
            raise ValueError("Model must be fit before predicting.")

        dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        return pd.DataFrame({
            "date": dates,
            "prediction": [self._last_value] * horizon,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """
        return {"name": self._name}


class SeasonalNaiveModel(BaseModel):
    """Seasonal naive forecasting model.

    This model predicts by repeating the last seasonal pattern.
    By default, it uses weekly seasonality (season_length=7).

    Attributes:
        _season_length: Number of periods in one seasonal cycle.
        _seasonal_values: The last season_length values from training.
        _last_date: The last date from training data.
    """

    def __init__(self, season_length: int = 7, name: str = "seasonal_naive") -> None:
        """Initialize the SeasonalNaiveModel.

        Args:
            season_length: Number of periods in one seasonal cycle.
            name: The name of the model.
        """
        super().__init__(name)
        self._season_length = season_length
        self._seasonal_values: np.ndarray | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_df: pd.DataFrame, target: str) -> Self:
        """Fit the model by storing the last season of values.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.

        Returns:
            Self for method chaining.
        """
        self._seasonal_values = train_df[target].iloc[-self._season_length:].values
        self._last_date = pd.Timestamp(train_df["date"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Generate predictions by repeating the seasonal pattern.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            DataFrame with 'date' and 'prediction' columns.
        """
        if self._seasonal_values is None or self._last_date is None:
            raise ValueError("Model must be fit before predicting.")

        dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        # Repeat the seasonal pattern to cover the horizon
        predictions = np.tile(self._seasonal_values, (horizon // self._season_length) + 1)
        predictions = predictions[:horizon]

        return pd.DataFrame({
            "date": dates,
            "prediction": predictions.tolist(),
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """
        params = super().get_params()
        params["season_length"] = self._season_length
        return params


class MovingAverageModel(BaseModel):
    """Moving average forecasting model.

    This model predicts by using the average of the last 'window' values.

    Attributes:
        _window: Number of periods to include in the moving average.
        _moving_average: The computed moving average value.
        _last_date: The last date from training data.
    """

    def __init__(self, window: int = 7, name: str = "moving_average") -> None:
        """Initialize the MovingAverageModel.

        Args:
            window: Number of periods for the moving average.
            name: The name of the model.
        """
        super().__init__(name)
        self._window = window
        self._moving_average: float | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_df: pd.DataFrame, target: str) -> Self:
        """Fit the model by computing the moving average.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.

        Returns:
            Self for method chaining.
        """
        self._moving_average = train_df[target].iloc[-self._window:].mean()
        self._last_date = pd.Timestamp(train_df["date"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Generate predictions using the moving average.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            DataFrame with 'date' and 'prediction' columns.
        """
        if self._moving_average is None or self._last_date is None:
            raise ValueError("Model must be fit before predicting.")

        dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        return pd.DataFrame({
            "date": dates,
            "prediction": [self._moving_average] * horizon,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """
        params = super().get_params()
        params["window"] = self._window
        return params
