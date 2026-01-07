"""Tree-based forecasting models (LightGBM, XGBoost).

This module provides tree-based machine learning models for time series
forecasting, including LightGBM and XGBoost implementations.
"""

from typing import Any, Self

import numpy as np
import pandas as pd

from volume_forecast.models.base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM-based forecasting model with lag and external features.

    This model uses LightGBM for gradient boosting regression with
    automatically generated lag features and optional external features.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth (-1 means no limit).
        learning_rate: Boosting learning rate.
        lags: List of lag periods for feature creation.
        external_features: List of external feature column names.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        lags: list[int] | None = None,
        external_features: list[str] | None = None,
        name: str = "lightgbm",
        **kwargs: Any,
    ) -> None:
        """Initialize the LightGBM model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth (-1 means no limit).
            learning_rate: Boosting learning rate.
            lags: List of lag periods for feature creation (default: [1, 7, 14]).
            external_features: List of external feature columns to use.
            name: Name of the model.
            **kwargs: Additional parameters passed to LGBMRegressor (e.g., num_leaves).
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lags = lags if lags is not None else [1, 7, 14]
        self.external_features = external_features or []
        self._model_kwargs = kwargs
        self._model: Any = None
        self._target: str | None = None
        self._train_data: pd.DataFrame | None = None
        self._last_train_date: pd.Timestamp | None = None
        self._feature_columns: list[str] = []

    def _create_lag_features(
        self, df: pd.DataFrame, target: str
    ) -> pd.DataFrame:
        """Create lag features from target column.

        Args:
            df: DataFrame with target column.
            target: Name of the target column.

        Returns:
            DataFrame with lag features added.
        """
        result = df.copy()
        for lag in self.lags:
            result[f"lag_{lag}"] = result[target].shift(lag)
        return result

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        feature_columns: list[str] | None = None,
    ) -> Self:
        """Fit the LightGBM model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            feature_columns: Optional list of feature columns. If None,
                lag features + external_features will be used.

        Returns:
            Self for method chaining.
        """
        self._target = target
        self._train_data = train_df.copy()
        self._last_train_date = pd.Timestamp(train_df["date"].iloc[-1])

        # Create lag features
        df_with_features = self._create_lag_features(train_df, target)
        lag_columns = [f"lag_{lag}" for lag in self.lags]

        # Determine feature columns
        if feature_columns is None:
            self._feature_columns = lag_columns + list(self.external_features)
        else:
            self._feature_columns = feature_columns

        # Validate external features exist
        for col in self.external_features:
            if col not in df_with_features.columns:
                raise ValueError(
                    f"External feature '{col}' not found in training data"
                )

        # Drop rows with NaN values (from lag features)
        df_clean = df_with_features.dropna(subset=self._feature_columns)

        X = df_clean[self._feature_columns].values
        y = df_clean[target].values

        # Import at runtime to avoid import errors if library not available
        from lightgbm import LGBMRegressor

        # Initialize and fit LightGBM model
        self._model = LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbose=-1,
            **self._model_kwargs,
        )
        self._model.fit(X, y)

        return self

    def predict(
        self,
        horizon: int = 7,
        future_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for future time periods.

        Args:
            horizon: Number of periods to forecast.
            future_df: Optional DataFrame with feature values for future dates.
                Must contain the date column and external feature columns if used.

        Returns:
            DataFrame with 'date' and 'prediction' columns.

        Raises:
            ValueError: If the model hasn't been fitted.
        """
        if self._model is None or self._train_data is None:
            raise ValueError("Model must be fitted before predicting.")
        if self._target is None or self._last_train_date is None:
            raise ValueError("Model must be fitted before predicting.")

        # Validate future_df if external features are used
        if self.external_features and future_df is None:
            raise ValueError(
                "future_df required when model uses external features"
            )
        if self.external_features and future_df is not None:
            if len(future_df) < horizon:
                raise ValueError(
                    f"future_df must have at least {horizon} rows for the forecast horizon"
                )
            for col in self.external_features:
                if col not in future_df.columns:
                    raise ValueError(
                        f"External feature '{col}' not found in future_df"
                    )

        # Generate prediction dates
        prediction_dates = pd.date_range(
            start=self._last_train_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        # Use historical values for lag features
        historical_values = self._train_data[self._target].tolist()

        predictions = []
        for step in range(horizon):
            # Build features for this prediction step
            feature_values = []

            # Add lag features
            for lag in self.lags:
                if len(historical_values) >= lag:
                    feature_values.append(historical_values[-lag])
                else:
                    # Use earliest available value if not enough history
                    feature_values.append(historical_values[0])

            # Add external features if present
            if self.external_features and future_df is not None:
                for col in self.external_features:
                    feature_values.append(future_df[col].iloc[step])

            # Predict next value
            X_pred = np.array([feature_values])
            pred = self._model.predict(X_pred)[0]
            predictions.append(pred)

            # Add prediction to history for next step
            historical_values.append(pred)

        return pd.DataFrame({
            "date": prediction_dates,
            "prediction": predictions,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """
        params = super().get_params()
        params.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "lags": self.lags,
            "external_features": self.external_features,
            **self._model_kwargs,
        })
        return params


class XGBoostModel(BaseModel):
    """XGBoost-based forecasting model with lag and external features.

    This model uses XGBoost for gradient boosting regression with
    automatically generated lag features and optional external features.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        lags: List of lag periods for feature creation.
        external_features: List of external feature column names.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        lags: list[int] | None = None,
        external_features: list[str] | None = None,
        name: str = "xgboost",
        **kwargs: Any,
    ) -> None:
        """Initialize the XGBoost model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            lags: List of lag periods for feature creation (default: [1, 7, 14]).
            external_features: List of external feature columns to use.
            name: Name of the model.
            **kwargs: Additional parameters passed to XGBRegressor (e.g., min_child_weight).
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lags = lags if lags is not None else [1, 7, 14]
        self.external_features = external_features or []
        self._model_kwargs = kwargs
        self._model: Any = None
        self._target: str | None = None
        self._train_data: pd.DataFrame | None = None
        self._last_train_date: pd.Timestamp | None = None
        self._feature_columns: list[str] = []

    def _create_lag_features(
        self, df: pd.DataFrame, target: str
    ) -> pd.DataFrame:
        """Create lag features from target column.

        Args:
            df: DataFrame with target column.
            target: Name of the target column.

        Returns:
            DataFrame with lag features added.
        """
        result = df.copy()
        for lag in self.lags:
            result[f"lag_{lag}"] = result[target].shift(lag)
        return result

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        feature_columns: list[str] | None = None,
    ) -> Self:
        """Fit the XGBoost model to training data.

        Args:
            train_df: Training DataFrame containing features and target.
            target: Name of the target column.
            feature_columns: Optional list of feature columns. If None,
                lag features + external_features will be used.

        Returns:
            Self for method chaining.
        """
        self._target = target
        self._train_data = train_df.copy()
        self._last_train_date = pd.Timestamp(train_df["date"].iloc[-1])

        # Create lag features
        df_with_features = self._create_lag_features(train_df, target)
        lag_columns = [f"lag_{lag}" for lag in self.lags]

        # Determine feature columns
        if feature_columns is None:
            self._feature_columns = lag_columns + list(self.external_features)
        else:
            self._feature_columns = feature_columns

        # Validate external features exist
        for col in self.external_features:
            if col not in df_with_features.columns:
                raise ValueError(
                    f"External feature '{col}' not found in training data"
                )

        # Drop rows with NaN values (from lag features)
        df_clean = df_with_features.dropna(subset=self._feature_columns)

        X = df_clean[self._feature_columns].values
        y = df_clean[target].values

        # Import at runtime to avoid import errors if library not available
        from xgboost import XGBRegressor

        # Initialize and fit XGBoost model
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=0,
            **self._model_kwargs,
        )
        self._model.fit(X, y)

        return self

    def predict(
        self,
        horizon: int = 7,
        future_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for future time periods.

        Args:
            horizon: Number of periods to forecast.
            future_df: Optional DataFrame with feature values for future dates.
                Must contain the date column and external feature columns if used.

        Returns:
            DataFrame with 'date' and 'prediction' columns.

        Raises:
            ValueError: If the model hasn't been fitted.
        """
        if self._model is None or self._train_data is None:
            raise ValueError("Model must be fitted before predicting.")
        if self._target is None or self._last_train_date is None:
            raise ValueError("Model must be fitted before predicting.")

        # Validate future_df if external features are used
        if self.external_features and future_df is None:
            raise ValueError(
                "future_df required when model uses external features"
            )
        if self.external_features and future_df is not None:
            if len(future_df) < horizon:
                raise ValueError(
                    f"future_df must have at least {horizon} rows for the forecast horizon"
                )
            for col in self.external_features:
                if col not in future_df.columns:
                    raise ValueError(
                        f"External feature '{col}' not found in future_df"
                    )

        # Generate prediction dates
        prediction_dates = pd.date_range(
            start=self._last_train_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        # Use historical values for lag features
        historical_values = self._train_data[self._target].tolist()

        predictions = []
        for step in range(horizon):
            # Build features for this prediction step
            feature_values = []

            # Add lag features
            for lag in self.lags:
                if len(historical_values) >= lag:
                    feature_values.append(historical_values[-lag])
                else:
                    # Use earliest available value if not enough history
                    feature_values.append(historical_values[0])

            # Add external features if present
            if self.external_features and future_df is not None:
                for col in self.external_features:
                    feature_values.append(future_df[col].iloc[step])

            # Predict next value
            X_pred = np.array([feature_values])
            pred = self._model.predict(X_pred)[0]
            predictions.append(pred)

            # Add prediction to history for next step
            historical_values.append(pred)

        return pd.DataFrame({
            "date": prediction_dates,
            "prediction": predictions,
        })

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """
        params = super().get_params()
        params.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "lags": self.lags,
            "external_features": self.external_features,
            **self._model_kwargs,
        })
        return params
