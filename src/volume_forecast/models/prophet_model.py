"""Prophet forecasting model wrapper.

This module provides a wrapper around Facebook Prophet for time series forecasting.
Prophet is a procedure for forecasting time series data that is robust to
missing data, shifts in the trend, and large outliers.
"""

import logging
from typing import Any, Self

import pandas as pd
from prophet import Prophet

from volume_forecast.models.base import BaseModel


class ProphetModel(BaseModel):
    """Prophet forecasting model using Facebook Prophet.

    This model wraps Facebook Prophet for time series forecasting with
    automatic seasonality detection and trend modeling.

    Attributes:
        _yearly_seasonality: Whether to include yearly seasonality.
        _weekly_seasonality: Whether to include weekly seasonality.
        _daily_seasonality: Whether to include daily seasonality.
        _model: Fitted Prophet model.
        _last_date: The last date from training data.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        name: str = "prophet",
    ) -> None:
        """Initialize the ProphetModel.

        Args:
            yearly_seasonality: Whether to include yearly seasonality.
            weekly_seasonality: Whether to include weekly seasonality.
            daily_seasonality: Whether to include daily seasonality.
            name: The name of the model.
        """
        super().__init__(name)
        self._yearly_seasonality = yearly_seasonality
        self._weekly_seasonality = weekly_seasonality
        self._daily_seasonality = daily_seasonality
        self._model: Prophet | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_df: pd.DataFrame, target: str, date_column: str = "date") -> Self:
        """Fit the Prophet model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            date_column: Name of the date column.

        Returns:
            Self for method chaining.
        """
        # Suppress Prophet logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        # Convert to Prophet format (ds, y columns)
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(train_df[date_column]),
            "y": train_df[target].values,
        })

        # Create and fit Prophet model
        self._model = Prophet(
            yearly_seasonality=self._yearly_seasonality,
            weekly_seasonality=self._weekly_seasonality,
            daily_seasonality=self._daily_seasonality,
        )
        self._model.fit(prophet_df)

        self._last_date = pd.Timestamp(train_df[date_column].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Generate forecasts using the fitted model.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            DataFrame with 'date' and 'prediction' columns.

        Raises:
            ValueError: If model hasn't been fit.
        """
        if self._model is None or self._last_date is None:
            raise ValueError("Model must be fit before predicting.")

        # Create future dataframe starting from day after last training date
        future_dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        future_df = pd.DataFrame({"ds": future_dates})

        # Generate forecast
        forecast = self._model.predict(future_df)

        return pd.DataFrame({
            "date": future_dates,
            "prediction": forecast["yhat"].values,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters including seasonality settings.
        """
        params = super().get_params()
        params["yearly_seasonality"] = self._yearly_seasonality
        params["weekly_seasonality"] = self._weekly_seasonality
        params["daily_seasonality"] = self._daily_seasonality
        return params
