"""Model comparison utilities for forecasting models.

This module provides utilities for comparing multiple forecasting models
using walk-forward validation and aggregating their performance metrics.
"""

import numpy as np
import pandas as pd

from volume_forecast.evaluation.validation import WalkForwardValidator
from volume_forecast.models.base import BaseModel


class ModelBenchmark:
    """Benchmark multiple forecasting models using walk-forward validation.

    This class provides functionality to:
    - Run validation for multiple models on the same dataset
    - Aggregate results into a comparison DataFrame
    - Identify the best performing model based on a chosen metric

    Attributes:
        _models: List of forecast models to compare.
        _validator: WalkForwardValidator instance for validation.
        _results_df: DataFrame containing benchmark results (None until benchmark runs).

    Example:
        >>> from volume_forecast.models.baselines import NaiveModel, MovingAverageModel
        >>> from volume_forecast.evaluation.validation import WalkForwardValidator
        >>>
        >>> models = [NaiveModel(), MovingAverageModel(window=7)]
        >>> validator = WalkForwardValidator(min_train_size=365, test_size=7)
        >>> benchmark = ModelBenchmark(models=models, validator=validator)
        >>>
        >>> results = benchmark.benchmark(df, target="daily_logins")
        >>> best_model = benchmark.get_best_model(metric="mae")
    """

    def __init__(
        self,
        models: list[BaseModel],
        validator: WalkForwardValidator,
    ) -> None:
        """Initialize the ModelBenchmark.

        Args:
            models: List of forecast models to compare. Each model should
                implement the BaseModel interface (fit and predict methods).
            validator: WalkForwardValidator instance for performing
                walk-forward validation on each model.
        """
        self._models = models
        self._validator = validator
        self._results_df: pd.DataFrame | None = None

    def benchmark(
        self,
        df: pd.DataFrame,
        target: str,
        date_column: str = "date",
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run benchmark comparison of all models.

        For each model, runs walk-forward validation and aggregates metrics
        (mean and standard deviation) across all folds.

        Args:
            df: DataFrame containing features and target.
            target: Name of the target column.
            date_column: Name of the date column.
            feature_columns: Optional list of feature columns to pass to models.
                If provided, future feature values will be passed to predict().

        Returns:
            DataFrame with columns:
            - model_name: Name of the model
            - mae_mean, mae_std: Mean Absolute Error statistics
            - rmse_mean, rmse_std: Root Mean Squared Error statistics
            - mape_mean, mape_std: Mean Absolute Percentage Error statistics
            - smape_mean, smape_std: Symmetric MAPE statistics
        """
        results_list = []

        for model in self._models:
            # Run validation for this model
            fold_results = self._validator.validate(
                model=model,
                df=df,
                target=target,
                date_column=date_column,
                feature_columns=feature_columns,
            )

            # Extract metrics from each fold
            mae_values = [r["metrics"]["mae"] for r in fold_results]
            rmse_values = [r["metrics"]["rmse"] for r in fold_results]
            mape_values = [r["metrics"]["mape"] for r in fold_results]
            smape_values = [r["metrics"]["smape"] for r in fold_results]

            # Compute aggregated statistics
            model_result = {
                "model_name": model.name,
                "mae_mean": np.mean(mae_values),
                "mae_std": np.std(mae_values),
                "rmse_mean": np.mean(rmse_values),
                "rmse_std": np.std(rmse_values),
                "mape_mean": np.mean(mape_values),
                "mape_std": np.std(mape_values),
                "smape_mean": np.mean(smape_values),
                "smape_std": np.std(smape_values),
            }

            results_list.append(model_result)

        self._results_df = pd.DataFrame(results_list)
        return self._results_df

    def get_best_model(self, metric: str = "mae") -> BaseModel:
        """Return the model with the best (lowest) mean metric value.

        Args:
            metric: The metric to use for comparison. Must be one of
                'mae', 'rmse', 'mape', or 'smape'.

        Returns:
            The BaseModel instance with the lowest mean value for the
            specified metric.

        Raises:
            ValueError: If benchmark has not been run yet, or if an
                invalid metric is specified.
        """
        if self._results_df is None:
            raise ValueError(
                "No benchmark results available. Run benchmark() first."
            )

        valid_metrics = ["mae", "rmse", "mape", "smape"]
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of {valid_metrics}."
            )

        metric_column = f"{metric}_mean"
        best_idx = self._results_df[metric_column].idxmin()
        best_model_name = self._results_df.loc[best_idx, "model_name"]

        # Find and return the model with the matching name
        for model in self._models:
            if model.name == best_model_name:
                return model

        # This should never happen if the benchmark was run correctly
        raise ValueError(f"Model '{best_model_name}' not found in models list.")

    def get_results_summary(self) -> pd.DataFrame:
        """Return the comparison DataFrame.

        Returns:
            DataFrame containing benchmark results for all models.

        Raises:
            ValueError: If benchmark has not been run yet.
        """
        if self._results_df is None:
            raise ValueError(
                "No benchmark results available. Run benchmark() first."
            )

        return self._results_df.copy()
