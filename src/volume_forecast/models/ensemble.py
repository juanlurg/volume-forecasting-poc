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
    Supports automatic inverse-MAE weighting or manual/equal weights.

    Attributes:
        _models: List of component models.
        _weights: List of weights for each model (sum to 1.0).
        _weighting: Weighting strategy ('inverse_mae', 'equal', or 'manual').
        _is_fitted: Whether the model has been fitted.
        _last_date: The last date from training data.
        _target: Target column name.
    """

    def __init__(
        self,
        models: list[BaseModel],
        weights: list[float] | None = None,
        weighting: str = "inverse_mae",
        name: str = "ensemble",
    ) -> None:
        """Initialize the EnsembleModel.

        Args:
            models: List of component models to ensemble.
            weights: Optional list of weights for each model. If provided,
                weighting is set to 'manual'. Weights must sum to 1.0.
            weighting: Weighting strategy when weights not provided:
                - 'inverse_mae': Compute weights from inverse training MAE (default)
                - 'equal': Equal weights (1/n) for each model
            name: The name of the model.

        Raises:
            ValueError: If weights are provided but don't sum to 1.0.
        """
        super().__init__(name)
        self._models = models

        if weights is not None:
            # Validate weights sum to 1.0
            if not np.isclose(sum(weights), 1.0):
                raise ValueError(f"Weights must sum to 1.0, but got {sum(weights):.4f}")
            self._weights = weights
            self._weighting = "manual"
        else:
            self._weights: list[float] = []
            self._weighting = weighting

        self._is_fitted = False
        self._last_date: pd.Timestamp | None = None
        self._target: str | None = None

    def fit(self, train_df: pd.DataFrame, target: str, **kwargs: Any) -> Self:
        """Fit all component models and compute ensemble weights.

        For inverse_mae weighting, computes weights based on each model's
        in-sample prediction accuracy (models with lower MAE get higher weight).

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            **kwargs: Additional keyword arguments passed to component models.

        Returns:
            Self for method chaining.
        """
        self._target = target
        mae_scores = []

        for model in self._models:
            model.fit(train_df, target, **kwargs)

            if self._weighting == "inverse_mae":
                # Calculate in-sample MAE for weight computation
                try:
                    predictions = model.predict(horizon=min(7, len(train_df) // 10))
                    n_preds = len(predictions)
                    y_true = train_df[target].iloc[-n_preds:].values

                    if "prediction" in predictions.columns:
                        y_pred = predictions["prediction"].values
                    elif "yhat" in predictions.columns:
                        y_pred = predictions["yhat"].values
                    else:
                        y_pred = predictions.iloc[:, -1].values

                    mae = np.mean(np.abs(y_true - y_pred))
                    mae_scores.append(max(mae, 1e-6))
                except Exception:
                    mae_scores.append(1e6)

        # Compute weights based on weighting strategy
        if self._weighting == "inverse_mae" and mae_scores:
            inverse_maes = [1.0 / mae for mae in mae_scores]
            total = sum(inverse_maes)
            self._weights = [inv / total for inv in inverse_maes]
        elif self._weighting == "equal":
            n = len(self._models)
            self._weights = [1.0 / n] * n
        # For 'manual', weights are already set in __init__

        self._last_date = pd.Timestamp(train_df["date"].iloc[-1])
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int = 7,
        future_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate predictions as weighted average of component models.

        Args:
            horizon: Number of periods to forecast.
            future_df: Optional DataFrame with feature values for future dates.
                Passed to component models that support external features.

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
            try:
                # Try passing future_df if model supports it
                if future_df is not None:
                    preds = model.predict(horizon=horizon, future_df=future_df)
                else:
                    preds = model.predict(horizon=horizon)
            except TypeError:
                # Model doesn't accept future_df parameter
                preds = model.predict(horizon=horizon)

            # Extract prediction values
            if "prediction" in preds.columns:
                pred_values = preds["prediction"].values
            elif "yhat" in preds.columns:
                pred_values = preds["yhat"].values
            else:
                pred_values = preds.iloc[:, -1].values

            all_predictions.append(pred_values)

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
        params["weighting"] = self._weighting
        params["model_names"] = [model.name for model in self._models]
        params["weights"] = self._weights
        return params

    def get_weights(self) -> dict[str, float]:
        """Return the computed weights for each component model.

        Returns:
            Dictionary mapping model names to their weights.
        """
        return {
            model.name: weight
            for model, weight in zip(self._models, self._weights)
        }
