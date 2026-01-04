"""Statistical forecasting models using statsmodels.

This module provides ARIMA/SARIMA wrappers using statsmodels SARIMAX
for time series forecasting.
"""

import warnings
from typing import Any, Self

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from volume_forecast.models.base import BaseModel


class ARIMAModel(BaseModel):
    """ARIMA/SARIMA forecasting model using statsmodels.

    This model wraps statsmodels SARIMAX for ARIMA and SARIMA forecasting.
    SARIMA adds seasonal components to handle periodic patterns.

    Attributes:
        _order: ARIMA order (p, d, q) for autoregressive, differencing, moving average.
        _seasonal_order: Seasonal order (P, D, Q, s) for seasonal components.
        _model: Fitted SARIMAX model result.
        _last_date: The last date from training data.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        name: str = "arima",
    ) -> None:
        """Initialize the ARIMAModel.

        Args:
            order: ARIMA order (p, d, q).
                - p: Number of autoregressive terms.
                - d: Degree of differencing.
                - q: Number of moving average terms.
            seasonal_order: Seasonal order (P, D, Q, s).
                - P: Seasonal autoregressive order.
                - D: Seasonal differencing order.
                - Q: Seasonal moving average order.
                - s: Seasonal period (e.g., 7 for weekly, 12 for monthly).
            name: The name of the model.
        """
        super().__init__(name)
        self._order = order
        self._seasonal_order = seasonal_order
        self._model: Any = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_df: pd.DataFrame, target: str) -> Self:
        """Fit the ARIMA/SARIMA model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.

        Returns:
            Self for method chaining.
        """
        # Extract target series
        y = train_df[target].values

        # Suppress convergence warnings - they are common and usually not critical
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Create and fit SARIMAX model
            model = SARIMAX(
                y,
                order=self._order,
                seasonal_order=self._seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._model = model.fit(disp=False)

        self._last_date = pd.Timestamp(train_df["date"].iloc[-1])
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

        # Generate forecast using get_forecast
        forecast = self._model.get_forecast(steps=horizon)
        predictions = forecast.predicted_mean

        # Generate dates
        dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        # Handle both pandas Series and numpy array return types
        pred_values = predictions.values if hasattr(predictions, "values") else predictions

        return pd.DataFrame({
            "date": dates,
            "prediction": pred_values,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters including order and seasonal_order.
        """
        params = super().get_params()
        params["order"] = self._order
        params["seasonal_order"] = self._seasonal_order
        return params
