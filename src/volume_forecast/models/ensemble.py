"""Ensemble forecasting model.

This module provides an ensemble model that combines predictions from
multiple component models using weighted averaging.
"""

from typing import Any, Self

import numpy as np
import pandas as pd

from volume_forecast.models.base import BaseModel


class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple forecasting models.

    This model computes a weighted average of predictions from component models.
    If no weights are provided, equal weights (1/n) are used.

    Attributes:
        _models: List of component models.
        _weights: List of weights for each model (sum to 1.0).
        _is_fitted: Whether the model has been fitted.
        _last_date: The last date from training data.
    """

    def __init__(
        self,
        models: list[BaseModel],
        weights: list[float] | None = None,
        name: str = "ensemble",
    ) -> None:
        """Initialize the EnsembleModel.

        Args:
            models: List of component models to ensemble.
            weights: Optional list of weights for each model. If None, equal
                weights (1/n) are used for each model. Weights must sum to 1.0.
            name: The name of the model.

        Raises:
            ValueError: If weights are provided but don't sum to 1.0.
        """
        super().__init__(name)
        self._models = models

        if weights is None:
            # Equal weights for all models
            n = len(models)
            self._weights = [1.0 / n] * n
        else:
            # Validate weights sum to 1.0
            if not np.isclose(sum(weights), 1.0):
                raise ValueError(f"Weights must sum to 1.0, but got {sum(weights):.4f}")
            self._weights = weights

        self._is_fitted = False
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_df: pd.DataFrame, target: str, **kwargs: Any) -> Self:
        """Fit all component models to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            **kwargs: Additional keyword arguments passed to component models.

        Returns:
            Self for method chaining.
        """
        for model in self._models:
            model.fit(train_df, target, **kwargs)

        self._last_date = pd.Timestamp(train_df["date"].iloc[-1])
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Generate predictions as weighted average of component models.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            DataFrame with 'date' and 'prediction' columns.

        Raises:
            ValueError: If model hasn't been fit.
        """
        if not self._is_fitted or self._last_date is None:
            raise ValueError("Model must be fit before predicting.")

        # Get predictions from each component model
        all_predictions = []
        for model in self._models:
            preds = model.predict(horizon=horizon)
            all_predictions.append(preds["prediction"].values)

        # Compute weighted average
        weighted_predictions = np.zeros(horizon)
        for weight, predictions in zip(self._weights, all_predictions, strict=True):
            weighted_predictions += weight * np.array(predictions)

        # Generate dates
        dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        return pd.DataFrame(
            {
                "date": dates,
                "prediction": weighted_predictions.tolist(),
            }
        )

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters including model names and weights.
        """
        params = super().get_params()
        params["model_names"] = [model.name for model in self._models]
        params["weights"] = self._weights
        return params
