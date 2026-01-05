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
    automatic seasonality detection, trend modeling, and optional external regressors.

    Attributes:
        _yearly_seasonality: Whether to include yearly seasonality.
        _weekly_seasonality: Whether to include weekly seasonality.
        _daily_seasonality: Whether to include daily seasonality.
        _regressors: List of external regressor column names.
        _model: Fitted Prophet model.
        _last_date: The last date from training data.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        regressors: list[str] | None = None,
        name: str = "prophet",
    ) -> None:
        """Initialize the ProphetModel.

        Args:
            yearly_seasonality: Whether to include yearly seasonality.
            weekly_seasonality: Whether to include weekly seasonality.
            daily_seasonality: Whether to include daily seasonality.
            regressors: List of external regressor column names.
            name: The name of the model.
        """
        super().__init__(name)
        self._yearly_seasonality = yearly_seasonality
        self._weekly_seasonality = weekly_seasonality
        self._daily_seasonality = daily_seasonality
        self._regressors = regressors or []
        self._model: Prophet | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        date_column: str = "date",
    ) -> Self:
        """Fit the Prophet model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            date_column: Name of the date column.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If any regressor column is not found in training data.
        """
        # Suppress Prophet logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        # Validate regressor columns exist in training data
        for reg in self._regressors:
            if reg not in train_df.columns:
                raise ValueError(f"Regressor '{reg}' not found in training data")

        # Convert to Prophet format (ds, y columns)
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(train_df[date_column]),
            "y": train_df[target].values,
        })

        # Add regressor columns to Prophet dataframe
        for reg in self._regressors:
            prophet_df[reg] = train_df[reg].values

        # Create Prophet model
        self._model = Prophet(
            yearly_seasonality=self._yearly_seasonality,
            weekly_seasonality=self._weekly_seasonality,
            daily_seasonality=self._daily_seasonality,
        )

        # Add regressors before fitting
        for reg in self._regressors:
            self._model.add_regressor(reg)

        # Fit the model
        self._model.fit(prophet_df)

        self._last_date = pd.Timestamp(train_df[date_column].iloc[-1])
        return self

    def predict(
        self,
        horizon: int = 7,
        future_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts using the fitted model.

        Args:
            horizon: Number of periods to forecast.
            future_df: DataFrame with regressor values for future dates.
                Required if regressors were used in training.

        Returns:
            DataFrame with 'date' and 'prediction' columns.

        Raises:
            ValueError: If model hasn't been fit, future_df is missing when
                regressors are used, or regressor columns are missing from future_df.
        """
        if self._model is None or self._last_date is None:
            raise ValueError("Model must be fit before predicting.")

        # Check if we need future regressor values
        if self._regressors and future_df is None:
            raise ValueError(
                "future_df required when model uses regressors. "
                f"Expected columns: {self._regressors}"
            )

        # Validate future_df has enough rows
        if future_df is not None and len(future_df) < horizon:
            raise ValueError(
                f"future_df must have at least {horizon} rows, got {len(future_df)}"
            )

        # Create future dataframe starting from day after last training date
        future_dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        prophet_future = pd.DataFrame({"ds": future_dates})

        # Add regressor values for future dates
        if self._regressors and future_df is not None:
            for reg in self._regressors:
                if reg not in future_df.columns:
                    raise ValueError(
                        f"Regressor '{reg}' not found in future_df"
                    )
                prophet_future[reg] = future_df[reg].values[:horizon]

        # Generate forecast
        forecast = self._model.predict(prophet_future)

        return pd.DataFrame({
            "date": future_dates,
            "prediction": forecast["yhat"].values,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters including seasonality settings and regressors.
        """
        params = super().get_params()
        params["yearly_seasonality"] = self._yearly_seasonality
        params["weekly_seasonality"] = self._weekly_seasonality
        params["daily_seasonality"] = self._daily_seasonality
        params["regressors"] = self._regressors
        return params
