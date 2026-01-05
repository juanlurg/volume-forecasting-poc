"""Walk-forward validation for time series forecasting.

Walk-forward validation is a technique for evaluating time series models that respects
temporal ordering. It uses an expanding window approach where:
1. The model is trained on historical data up to time t
2. Predictions are made for the next horizon period
3. The window expands and the process repeats

This ensures that models are only evaluated on truly out-of-sample data.
"""

from collections.abc import Generator
from typing import Any, Protocol

import numpy as np
import pandas as pd

from volume_forecast.evaluation.metrics import calculate_all_metrics


class ForecastModel(Protocol):
    """Protocol defining the interface for forecast models."""

    def fit(self, train_df: pd.DataFrame, target: str) -> "ForecastModel":
        """Fit the model to training data."""
        ...

    def predict(
        self, horizon: int, future_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Generate predictions.

        Args:
            horizon: Number of periods to forecast.
            future_df: Optional DataFrame with feature values for future dates.
        """
        ...


class WalkForwardValidator:
    """Walk-forward validation for time series models.

    Implements expanding window cross-validation where:
    - Training window starts at min_train_size and expands each fold
    - Test window is always test_size days
    - Each fold advances by step_size days

    This approach:
    - Respects temporal ordering (no future data leakage)
    - Captures full seasonality in training (default 365 days)
    - Matches the forecast horizon in test window (default 7 days)

    Attributes:
        min_train_size: Minimum number of training samples (default: 365 for yearly seasonality)
        test_size: Number of samples in test window (default: 7 for weekly forecast)
        step_size: How much to advance each fold (default: 7 for weekly steps)

    Example:
        >>> validator = WalkForwardValidator(min_train_size=365, test_size=7, step_size=7)
        >>> for train_df, test_df in validator.split(df, date_column="date"):
        ...     model.fit(train_df, "target")
        ...     predictions = model.predict(test_df, horizon=7)
    """

    def __init__(
        self,
        min_train_size: int = 365,
        test_size: int = 7,
        step_size: int = 7,
    ) -> None:
        """Initialize the walk-forward validator.

        Args:
            min_train_size: Minimum training window size. Should be at least
                one full seasonal cycle (365 for yearly seasonality).
            test_size: Test window size. Should match the forecast horizon.
            step_size: How much to advance the window each fold.
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size

    def get_n_splits(self, df: pd.DataFrame, date_column: str = "date") -> int:
        """Calculate the number of folds for the given data.

        Args:
            df: DataFrame containing the time series data.
            date_column: Name of the date column.

        Returns:
            Number of folds that can be generated from the data.
        """
        n_samples = len(df)
        min_required = self.min_train_size + self.test_size

        if n_samples < min_required:
            return 0

        # Calculate number of complete folds
        # First fold ends at min_train_size + test_size
        # Each subsequent fold adds step_size to the end
        remaining_after_first = n_samples - min_required
        additional_folds = remaining_after_first // self.step_size

        return 1 + additional_folds

    def split(
        self, df: pd.DataFrame, date_column: str = "date"
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Generate train/test splits for walk-forward validation.

        Uses expanding window approach where training set grows with each fold
        while test window remains fixed size.

        Args:
            df: DataFrame containing the time series data.
            date_column: Name of the date column.

        Yields:
            Tuple of (train_df, test_df) for each fold.
        """
        # Sort by date to ensure chronological order
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        n_samples = len(df_sorted)

        n_splits = self.get_n_splits(df_sorted, date_column)

        for fold_idx in range(n_splits):
            # Training end index (expanding window)
            train_end = self.min_train_size + (fold_idx * self.step_size)

            # Test window
            test_start = train_end
            test_end = test_start + self.test_size

            # Ensure we don't exceed data bounds
            if test_end > n_samples:
                break

            train_df = df_sorted.iloc[:train_end].copy()
            test_df = df_sorted.iloc[test_start:test_end].copy()

            yield train_df, test_df

    def validate(
        self,
        model: ForecastModel,
        df: pd.DataFrame,
        target: str,
        date_column: str = "date",
        feature_columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run walk-forward validation on a model.

        For each fold:
        1. Fit the model on training data
        2. Generate predictions for the test period
        3. Calculate evaluation metrics

        Args:
            model: A forecast model implementing fit() and predict().
            df: DataFrame containing features and target.
            target: Name of the target column.
            date_column: Name of the date column.
            feature_columns: Optional list of feature columns to pass to model.
                If provided, future feature values will be passed to predict().

        Returns:
            List of dictionaries, one per fold, containing:
            - fold_id: Index of the fold (0-based)
            - metrics: Dictionary of evaluation metrics (mae, rmse, mape, smape)
            - predictions: DataFrame with actual values and predictions
        """
        results: list[dict[str, Any]] = []

        for fold_id, (train_df, test_df) in enumerate(self.split(df, date_column=date_column)):
            # Fit model on training data
            model.fit(train_df, target)

            # Prepare future_df if feature columns are specified
            future_df = None
            if feature_columns:
                future_df = test_df[[date_column] + list(feature_columns)].copy()

            # Generate predictions - try with future_df first, fall back to without
            try:
                if future_df is not None:
                    predictions_df = model.predict(horizon=self.test_size, future_df=future_df)
                else:
                    predictions_df = model.predict(horizon=self.test_size)
            except TypeError:
                # Model doesn't accept future_df parameter
                predictions_df = model.predict(horizon=self.test_size)

            # Get actual values
            y_true = test_df[target].values

            # Get predicted values (handle different column naming conventions)
            if "prediction" in predictions_df.columns:
                y_pred = predictions_df["prediction"].values
            elif "yhat" in predictions_df.columns:
                y_pred = predictions_df["yhat"].values
            elif target in predictions_df.columns:
                y_pred = predictions_df[target].values
            else:
                # Assume first numeric column is predictions
                numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_pred = predictions_df[numeric_cols[0]].values
                else:
                    raise ValueError(
                        f"Could not find prediction column in {predictions_df.columns.tolist()}"
                    )

            # Calculate metrics
            metrics = calculate_all_metrics(y_true, y_pred)

            # Build result dictionary
            result = {
                "fold_id": fold_id,
                "metrics": metrics,
                "predictions": pd.DataFrame(
                    {
                        date_column: test_df[date_column].values,
                        "actual": y_true,
                        "predicted": y_pred,
                    }
                ),
            }

            results.append(result)

        return results
