"""Tests for models module."""

import pandas as pd
import pytest


class TestBaseModel:
    """Test suite for base model interface."""

    @pytest.fixture
    def sample_train_df(self) -> pd.DataFrame:
        """Create sample training DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "daily_logins": range(100, 130),
            "feature_1": range(30),
        })

    @pytest.fixture
    def sample_predict_df(self) -> pd.DataFrame:
        """Create sample DataFrame for prediction."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-31", periods=7, freq="D"),
            "feature_1": range(7),
        })

    def test_fit_returns_self(self, sample_train_df: pd.DataFrame) -> None:
        """fit() should return self for chaining."""
        from volume_forecast.models.base import BaseModel

        class DummyModel(BaseModel):
            def fit(self, train_df: pd.DataFrame, target: str) -> "DummyModel":
                return self

            def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
                return pd.DataFrame({"forecast": [0] * horizon})

        model = DummyModel(name="dummy")
        result = model.fit(sample_train_df, target="daily_logins")

        assert result is model

    def test_abstract_predict_raises(self, sample_predict_df: pd.DataFrame) -> None:
        """predict() should be abstract and raise if not implemented."""
        from volume_forecast.models.base import BaseModel

        # Cannot instantiate BaseModel directly without implementing abstract methods
        with pytest.raises(TypeError, match="abstract"):
            BaseModel(name="abstract_test")  # type: ignore[abstract]

    def test_get_params_returns_dict(self) -> None:
        """get_params() should return a dict."""
        from volume_forecast.models.base import BaseModel

        class DummyModel(BaseModel):
            def __init__(self, name: str, learning_rate: float = 0.01) -> None:
                super().__init__(name)
                self.learning_rate = learning_rate

            def fit(self, train_df: pd.DataFrame, target: str) -> "DummyModel":
                return self

            def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
                return pd.DataFrame({"forecast": [0] * horizon})

            def get_params(self) -> dict:
                params = super().get_params()
                params["learning_rate"] = self.learning_rate
                return params

        model = DummyModel(name="dummy", learning_rate=0.05)
        params = model.get_params()

        assert isinstance(params, dict)
        assert "name" in params
        assert params["name"] == "dummy"
        assert params["learning_rate"] == 0.05

    def test_name_property(self) -> None:
        """name property should return model name."""
        from volume_forecast.models.base import BaseModel

        class DummyModel(BaseModel):
            def fit(self, train_df: pd.DataFrame, target: str) -> "DummyModel":
                return self

            def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
                return pd.DataFrame({"forecast": [0] * horizon})

        model = DummyModel(name="test_model")
        assert model.name == "test_model"

    def test_init_stores_name(self) -> None:
        """__init__ should store the model name."""
        from volume_forecast.models.base import BaseModel

        class DummyModel(BaseModel):
            def fit(self, train_df: pd.DataFrame, target: str) -> "DummyModel":
                return self

            def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
                return pd.DataFrame({"forecast": [0] * horizon})

        model = DummyModel(name="my_model")
        assert model.name == "my_model"

    def test_fit_is_abstract(self) -> None:
        """fit() should be abstract."""
        from volume_forecast.models.base import BaseModel

        # Try to create a class that only implements predict (not fit)
        class PartialModel(BaseModel):
            def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
                return pd.DataFrame({"forecast": [0] * horizon})

        with pytest.raises(TypeError, match="abstract"):
            PartialModel(name="partial")  # type: ignore[abstract]

    def test_predict_is_abstract(self) -> None:
        """predict() should be abstract."""
        from volume_forecast.models.base import BaseModel

        # Try to create a class that only implements fit (not predict)
        class PartialModel(BaseModel):
            def fit(self, train_df: pd.DataFrame, target: str) -> "PartialModel":
                return self

        with pytest.raises(TypeError, match="abstract"):
            PartialModel(name="partial")  # type: ignore[abstract]


class TestBaselineModels:
    """Test suite for baseline forecasting models."""

    @pytest.fixture
    def sample_train_df(self) -> pd.DataFrame:
        """Create sample training DataFrame with known values."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=14, freq="D"),
            "daily_logins": [100, 110, 120, 130, 140, 150, 160,  # Week 1
                             200, 210, 220, 230, 240, 250, 260],  # Week 2
        })

    def test_naive_model_predicts_last_value(self, sample_train_df: pd.DataFrame) -> None:
        """NaiveModel should predict the last observed value."""
        from volume_forecast.models.baselines import NaiveModel

        model = NaiveModel()
        model.fit(sample_train_df, target="daily_logins")
        predictions = model.predict(horizon=3)

        # Should predict 260 (last value) for all 3 days
        assert len(predictions) == 3
        assert "date" in predictions.columns
        assert "prediction" in predictions.columns
        assert all(predictions["prediction"] == 260)

    def test_seasonal_naive_predicts_same_day_last_week(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """SeasonalNaiveModel should predict value from 7 days ago."""
        from volume_forecast.models.baselines import SeasonalNaiveModel

        model = SeasonalNaiveModel(season_length=7)
        model.fit(sample_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        # Should repeat the last week's pattern: 200, 210, 220, 230, 240, 250, 260
        expected = [200, 210, 220, 230, 240, 250, 260]
        assert len(predictions) == 7
        assert "date" in predictions.columns
        assert "prediction" in predictions.columns
        assert list(predictions["prediction"]) == expected

    def test_moving_average_predicts_mean(self, sample_train_df: pd.DataFrame) -> None:
        """MovingAverageModel should predict the rolling mean."""
        from volume_forecast.models.baselines import MovingAverageModel

        model = MovingAverageModel(window=7)
        model.fit(sample_train_df, target="daily_logins")
        predictions = model.predict(horizon=3)

        # Mean of last 7 values: (200+210+220+230+240+250+260)/7 = 230
        expected_mean = 230.0
        assert len(predictions) == 3
        assert "date" in predictions.columns
        assert "prediction" in predictions.columns
        assert all(predictions["prediction"] == expected_mean)

    def test_naive_model_extends_base_model(self) -> None:
        """NaiveModel should extend BaseModel."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.baselines import NaiveModel

        model = NaiveModel()
        assert isinstance(model, BaseModel)

    def test_seasonal_naive_extends_base_model(self) -> None:
        """SeasonalNaiveModel should extend BaseModel."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.baselines import SeasonalNaiveModel

        model = SeasonalNaiveModel()
        assert isinstance(model, BaseModel)

    def test_moving_average_extends_base_model(self) -> None:
        """MovingAverageModel should extend BaseModel."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.baselines import MovingAverageModel

        model = MovingAverageModel()
        assert isinstance(model, BaseModel)

    def test_naive_model_fit_returns_self(self, sample_train_df: pd.DataFrame) -> None:
        """NaiveModel.fit() should return self for chaining."""
        from volume_forecast.models.baselines import NaiveModel

        model = NaiveModel()
        result = model.fit(sample_train_df, target="daily_logins")
        assert result is model

    def test_seasonal_naive_fit_returns_self(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """SeasonalNaiveModel.fit() should return self for chaining."""
        from volume_forecast.models.baselines import SeasonalNaiveModel

        model = SeasonalNaiveModel()
        result = model.fit(sample_train_df, target="daily_logins")
        assert result is model

    def test_moving_average_fit_returns_self(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """MovingAverageModel.fit() should return self for chaining."""
        from volume_forecast.models.baselines import MovingAverageModel

        model = MovingAverageModel()
        result = model.fit(sample_train_df, target="daily_logins")
        assert result is model

    def test_predictions_have_correct_date_range(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """Predictions should start from day after last training date."""
        from volume_forecast.models.baselines import NaiveModel

        model = NaiveModel()
        model.fit(sample_train_df, target="daily_logins")
        predictions = model.predict(horizon=3)

        # Last training date is 2024-01-14, predictions should start 2024-01-15
        expected_dates = pd.date_range("2024-01-15", periods=3, freq="D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(predictions["date"]),
            expected_dates,
            check_names=False,
        )

    def test_seasonal_naive_wraps_around_for_long_horizon(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """SeasonalNaiveModel should repeat pattern for horizons > season_length."""
        from volume_forecast.models.baselines import SeasonalNaiveModel

        model = SeasonalNaiveModel(season_length=7)
        model.fit(sample_train_df, target="daily_logins")
        predictions = model.predict(horizon=10)

        # Should repeat pattern: 200,210,220,230,240,250,260, then 200,210,220
        expected = [200, 210, 220, 230, 240, 250, 260, 200, 210, 220]
        assert list(predictions["prediction"]) == expected


class TestStatisticalModels:
    """Test suite for statistical forecasting models (ARIMA/SARIMA)."""

    @pytest.fixture
    def simple_train_df(self) -> pd.DataFrame:
        """Create simple training DataFrame with linear trend for fast fitting."""
        # Use simple data that ARIMA can fit quickly
        import numpy as np

        np.random.seed(42)
        n = 50  # Enough data for ARIMA but fast
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        # Simple linear trend with small noise
        values = 100 + np.arange(n) * 2 + np.random.normal(0, 2, n)
        return pd.DataFrame({
            "date": dates,
            "daily_logins": values,
        })

    @pytest.fixture
    def seasonal_train_df(self) -> pd.DataFrame:
        """Create training DataFrame with weekly seasonality for SARIMA."""
        import numpy as np

        np.random.seed(42)
        n = 60  # At least 8 weeks of data
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        # Weekly pattern: higher on weekends
        seasonal = np.array([0, 0, 0, 0, 10, 20, 15] * (n // 7 + 1))[:n]
        values = 100 + seasonal + np.random.normal(0, 2, n)
        return pd.DataFrame({
            "date": dates,
            "daily_logins": values,
        })

    def test_arima_model_extends_base_model(self) -> None:
        """ARIMAModel should extend BaseModel."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel()
        assert isinstance(model, BaseModel)

    def test_arima_model_fit_returns_self(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """fit() should return self for chaining."""
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel(order=(1, 0, 0))  # Simple AR(1) for speed
        result = model.fit(simple_train_df, target="daily_logins")
        assert result is model

    def test_arima_model_predict_returns_dataframe(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """predict() should return DataFrame with date and prediction columns."""
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel(order=(1, 0, 0))
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        assert isinstance(predictions, pd.DataFrame)
        assert "date" in predictions.columns
        assert "prediction" in predictions.columns

    def test_arima_model_prediction_length(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """predictions should have correct length (horizon)."""
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel(order=(1, 0, 0))
        model.fit(simple_train_df, target="daily_logins")

        for horizon in [3, 7, 14]:
            predictions = model.predict(horizon=horizon)
            assert len(predictions) == horizon

    def test_arima_model_prediction_dates_are_sequential(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """Prediction dates should start after training data and be sequential."""
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel(order=(1, 0, 0))
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        # First prediction should be day after last training date
        last_train_date = simple_train_df["date"].iloc[-1]
        expected_start = pd.Timestamp(last_train_date) + pd.Timedelta(days=1)
        assert pd.Timestamp(predictions["date"].iloc[0]) == expected_start

        # Dates should be sequential
        expected_dates = pd.date_range(start=expected_start, periods=7, freq="D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(predictions["date"]),
            expected_dates,
            check_names=False,
        )

    def test_arima_model_get_params_returns_dict(self) -> None:
        """get_params() should return a dict with order and seasonal_order."""
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel(order=(2, 1, 1), seasonal_order=(1, 0, 1, 7))
        params = model.get_params()

        assert isinstance(params, dict)
        assert "name" in params
        assert "order" in params
        assert "seasonal_order" in params
        assert params["order"] == (2, 1, 1)
        assert params["seasonal_order"] == (1, 0, 1, 7)

    def test_arima_model_default_params(self) -> None:
        """ARIMAModel should have sensible default parameters."""
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel()
        params = model.get_params()

        assert params["order"] == (1, 1, 1)
        assert params["seasonal_order"] == (0, 0, 0, 0)

    def test_arima_model_raises_before_fit(self) -> None:
        """predict() should raise if model hasn't been fit."""
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel()
        with pytest.raises(ValueError, match="fit"):
            model.predict(horizon=7)

    @pytest.mark.slow
    def test_sarima_model_with_seasonality(
        self, seasonal_train_df: pd.DataFrame
    ) -> None:
        """SARIMA should work with seasonal parameters."""
        from volume_forecast.models.statistical import ARIMAModel

        # SARIMA(1,0,0)(1,0,0,7) - weekly seasonality
        model = ARIMAModel(order=(1, 0, 0), seasonal_order=(1, 0, 0, 7))
        model.fit(seasonal_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        assert len(predictions) == 7
        assert "prediction" in predictions.columns
        # Predictions should be reasonable (positive values)
        assert all(predictions["prediction"] > 0)

    def test_arima_model_predictions_are_numeric(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """Predictions should be numeric values."""
        from volume_forecast.models.statistical import ARIMAModel

        model = ARIMAModel(order=(1, 0, 0))
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        # Check predictions are numeric
        assert predictions["prediction"].dtype in [float, "float64", "float32"]
        # Check no NaN values
        assert not predictions["prediction"].isna().any()


def _lightgbm_available() -> bool:
    """Check if LightGBM can be imported successfully."""
    try:
        from lightgbm import LGBMRegressor  # noqa: F401
        return True
    except (ImportError, FileNotFoundError, OSError):
        return False


lightgbm_available = pytest.mark.skipif(
    not _lightgbm_available(),
    reason="LightGBM not available (missing runtime dependencies)",
)


class TestTreeModels:
    """Test suite for tree-based forecasting models (LightGBM, XGBoost)."""

    @pytest.fixture
    def simple_train_df(self) -> pd.DataFrame:
        """Create simple training DataFrame with enough history for lag features."""
        import numpy as np

        np.random.seed(42)
        n = 60  # Enough data for lag features (up to 14 days)
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        # Simple linear trend with small noise
        values = 100 + np.arange(n) * 2 + np.random.normal(0, 5, n)
        return pd.DataFrame({
            "date": dates,
            "daily_logins": values,
        })

    @pytest.fixture
    def predict_df(self, simple_train_df: pd.DataFrame) -> pd.DataFrame:
        """Create DataFrame for prediction starting after training data."""
        last_date = simple_train_df["date"].iloc[-1]
        return pd.DataFrame({
            "date": pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq="D"),
        })

    @lightgbm_available
    def test_lightgbm_model_extends_base_model(self) -> None:
        """LightGBMModel should extend BaseModel."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel()
        assert isinstance(model, BaseModel)

    @lightgbm_available
    def test_lightgbm_model_fit_returns_self(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """fit() should return self for chaining."""
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel()
        result = model.fit(simple_train_df, target="daily_logins")
        assert result is model

    @lightgbm_available
    def test_lightgbm_model_predict_returns_dataframe(
        self, simple_train_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> None:
        """predict() should return DataFrame with date and prediction columns."""
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(predict_df, horizon=7)

        assert isinstance(predictions, pd.DataFrame)
        assert "date" in predictions.columns
        assert "prediction" in predictions.columns
        assert len(predictions) == 7

    @lightgbm_available
    def test_lightgbm_model_prediction_dates_are_sequential(
        self, simple_train_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> None:
        """Prediction dates should start after training data and be sequential."""
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(predict_df, horizon=7)

        # First prediction should be day after last training date
        last_train_date = simple_train_df["date"].iloc[-1]
        expected_start = pd.Timestamp(last_train_date) + pd.Timedelta(days=1)
        assert pd.Timestamp(predictions["date"].iloc[0]) == expected_start

        # Dates should be sequential
        expected_dates = pd.date_range(start=expected_start, periods=7, freq="D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(predictions["date"]),
            expected_dates,
            check_names=False,
        )

    @lightgbm_available
    def test_lightgbm_model_get_params_returns_dict(self) -> None:
        """get_params() should return a dict with model parameters."""
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel(n_estimators=200, max_depth=5, learning_rate=0.05)
        params = model.get_params()

        assert isinstance(params, dict)
        assert "name" in params
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "lags" in params
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 5
        assert params["learning_rate"] == 0.05

    @lightgbm_available
    def test_lightgbm_model_default_params(self) -> None:
        """LightGBMModel should have sensible default parameters."""
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel()
        params = model.get_params()

        assert params["name"] == "lightgbm"
        assert params["n_estimators"] == 100
        assert params["max_depth"] == -1
        assert params["learning_rate"] == 0.1
        assert params["lags"] == [1, 7, 14]

    @lightgbm_available
    def test_lightgbm_model_raises_before_fit(self, predict_df: pd.DataFrame) -> None:
        """predict() should raise if model hasn't been fit."""
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel()
        with pytest.raises(ValueError, match="fit"):
            model.predict(predict_df, horizon=7)

    @lightgbm_available
    def test_lightgbm_model_predictions_are_numeric(
        self, simple_train_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> None:
        """Predictions should be numeric values."""
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(predict_df, horizon=7)

        # Check predictions are numeric
        assert predictions["prediction"].dtype in [float, "float64", "float32"]
        # Check no NaN values
        assert not predictions["prediction"].isna().any()

    def test_xgboost_model_extends_base_model(self) -> None:
        """XGBoostModel should extend BaseModel."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel()
        assert isinstance(model, BaseModel)

    def test_xgboost_model_fit_returns_self(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """fit() should return self for chaining."""
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel()
        result = model.fit(simple_train_df, target="daily_logins")
        assert result is model

    def test_xgboost_model_predict_returns_dataframe(
        self, simple_train_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> None:
        """predict() should return DataFrame with date and prediction columns."""
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(predict_df, horizon=7)

        assert isinstance(predictions, pd.DataFrame)
        assert "date" in predictions.columns
        assert "prediction" in predictions.columns
        assert len(predictions) == 7

    def test_xgboost_model_prediction_dates_are_sequential(
        self, simple_train_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> None:
        """Prediction dates should start after training data and be sequential."""
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(predict_df, horizon=7)

        # First prediction should be day after last training date
        last_train_date = simple_train_df["date"].iloc[-1]
        expected_start = pd.Timestamp(last_train_date) + pd.Timedelta(days=1)
        assert pd.Timestamp(predictions["date"].iloc[0]) == expected_start

        # Dates should be sequential
        expected_dates = pd.date_range(start=expected_start, periods=7, freq="D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(predictions["date"]),
            expected_dates,
            check_names=False,
        )

    def test_xgboost_model_get_params_returns_dict(self) -> None:
        """get_params() should return a dict with model parameters."""
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel(n_estimators=200, max_depth=8, learning_rate=0.05)
        params = model.get_params()

        assert isinstance(params, dict)
        assert "name" in params
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "lags" in params
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 8
        assert params["learning_rate"] == 0.05

    def test_xgboost_model_default_params(self) -> None:
        """XGBoostModel should have sensible default parameters."""
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel()
        params = model.get_params()

        assert params["name"] == "xgboost"
        assert params["n_estimators"] == 100
        assert params["max_depth"] == 6
        assert params["learning_rate"] == 0.1
        assert params["lags"] == [1, 7, 14]

    def test_xgboost_model_raises_before_fit(self, predict_df: pd.DataFrame) -> None:
        """predict() should raise if model hasn't been fit."""
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel()
        with pytest.raises(ValueError, match="fit"):
            model.predict(predict_df, horizon=7)

    def test_xgboost_model_predictions_are_numeric(
        self, simple_train_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> None:
        """Predictions should be numeric values."""
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(predict_df, horizon=7)

        # Check predictions are numeric
        assert predictions["prediction"].dtype in [float, "float64", "float32"]
        # Check no NaN values
        assert not predictions["prediction"].isna().any()

    @lightgbm_available
    def test_lightgbm_with_custom_lags(
        self, simple_train_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> None:
        """LightGBM should work with custom lag settings."""
        from volume_forecast.models.tree_models import LightGBMModel

        model = LightGBMModel(lags=[1, 2, 3])
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(predict_df, horizon=7)

        assert len(predictions) == 7
        params = model.get_params()
        assert params["lags"] == [1, 2, 3]

    def test_xgboost_with_custom_lags(
        self, simple_train_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> None:
        """XGBoost should work with custom lag settings."""
        from volume_forecast.models.tree_models import XGBoostModel

        model = XGBoostModel(lags=[1, 2, 3])
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(predict_df, horizon=7)

        assert len(predictions) == 7
        params = model.get_params()
        assert params["lags"] == [1, 2, 3]


class TestProphetModel:
    """Test suite for Prophet forecasting model."""

    @pytest.fixture
    def simple_train_df(self) -> pd.DataFrame:
        """Create simple training DataFrame for Prophet (60-90 days for faster tests)."""
        import numpy as np

        np.random.seed(42)
        n = 60  # Smaller dataset for faster Prophet fitting
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        # Simple linear trend with small noise
        values = 100 + np.arange(n) * 2 + np.random.normal(0, 5, n)
        return pd.DataFrame({
            "date": dates,
            "daily_logins": values,
        })

    @pytest.fixture
    def seasonal_train_df(self) -> pd.DataFrame:
        """Create training DataFrame with weekly seasonality for Prophet."""
        import numpy as np

        np.random.seed(42)
        n = 90  # 12+ weeks of data for seasonality
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        # Weekly pattern: higher on weekends
        seasonal = np.array([0, 0, 0, 0, 10, 20, 15] * (n // 7 + 1))[:n]
        values = 100 + seasonal + np.random.normal(0, 3, n)
        return pd.DataFrame({
            "date": dates,
            "daily_logins": values,
        })

    def test_prophet_model_extends_base_model(self) -> None:
        """ProphetModel should extend BaseModel."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel()
        assert isinstance(model, BaseModel)

    @pytest.mark.slow
    def test_prophet_model_fit_returns_self(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """fit() should return self for chaining."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel()
        result = model.fit(simple_train_df, target="daily_logins")
        assert result is model

    @pytest.mark.slow
    def test_prophet_model_predict_returns_dataframe(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """predict() should return DataFrame with date and prediction columns."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        assert isinstance(predictions, pd.DataFrame)
        assert "date" in predictions.columns
        assert "prediction" in predictions.columns

    @pytest.mark.slow
    def test_prophet_model_prediction_length(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """predictions should have correct length (horizon)."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel()
        model.fit(simple_train_df, target="daily_logins")

        for horizon in [3, 7, 14]:
            predictions = model.predict(horizon=horizon)
            assert len(predictions) == horizon

    @pytest.mark.slow
    def test_prophet_model_prediction_dates_are_sequential(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """Prediction dates should start after training data and be sequential."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        # First prediction should be day after last training date
        last_train_date = simple_train_df["date"].iloc[-1]
        expected_start = pd.Timestamp(last_train_date) + pd.Timedelta(days=1)
        assert pd.Timestamp(predictions["date"].iloc[0]) == expected_start

        # Dates should be sequential
        expected_dates = pd.date_range(start=expected_start, periods=7, freq="D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(predictions["date"]),
            expected_dates,
            check_names=False,
        )

    def test_prophet_model_get_params_returns_dict(self) -> None:
        """get_params() should return a dict with seasonality settings."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        params = model.get_params()

        assert isinstance(params, dict)
        assert "name" in params
        assert "yearly_seasonality" in params
        assert "weekly_seasonality" in params
        assert "daily_seasonality" in params
        assert params["yearly_seasonality"] is True
        assert params["weekly_seasonality"] is True
        assert params["daily_seasonality"] is False

    def test_prophet_model_default_params(self) -> None:
        """ProphetModel should have sensible default parameters."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel()
        params = model.get_params()

        assert params["name"] == "prophet"
        assert params["yearly_seasonality"] is True
        assert params["weekly_seasonality"] is True
        assert params["daily_seasonality"] is False

    def test_prophet_model_raises_before_fit(self) -> None:
        """predict() should raise if model hasn't been fit."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel()
        with pytest.raises(ValueError, match="fit"):
            model.predict(horizon=7)

    @pytest.mark.slow
    def test_prophet_model_predictions_are_numeric(
        self, simple_train_df: pd.DataFrame
    ) -> None:
        """Predictions should be numeric values."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel()
        model.fit(simple_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        # Check predictions are numeric
        assert predictions["prediction"].dtype in [float, "float64", "float32"]
        # Check no NaN values
        assert not predictions["prediction"].isna().any()

    @pytest.mark.slow
    def test_prophet_model_with_custom_seasonality(
        self, seasonal_train_df: pd.DataFrame
    ) -> None:
        """Prophet should work with custom seasonality settings."""
        from volume_forecast.models.prophet_model import ProphetModel

        model = ProphetModel(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(seasonal_train_df, target="daily_logins")
        predictions = model.predict(horizon=7)

        assert len(predictions) == 7
        assert "prediction" in predictions.columns
        # Predictions should be reasonable (positive values)
        assert all(predictions["prediction"] > 0)


class TestEnsembleModel:
    """Test suite for ensemble forecasting model."""

    @pytest.fixture
    def sample_train_df(self) -> pd.DataFrame:
        """Create sample training DataFrame with known values."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=14, freq="D"),
            "daily_logins": [100, 110, 120, 130, 140, 150, 160,  # Week 1
                             200, 210, 220, 230, 240, 250, 260],  # Week 2
        })

    def test_ensemble_model_extends_base_model(self) -> None:
        """EnsembleModel should extend BaseModel."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        ensemble = EnsembleModel(models=[naive, ma])
        assert isinstance(ensemble, BaseModel)

    def test_ensemble_model_fit_returns_self(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """fit() should return self for chaining."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        ensemble = EnsembleModel(models=[naive, ma])
        result = ensemble.fit(sample_train_df, target="daily_logins")
        assert result is ensemble

    def test_ensemble_model_predict_returns_dataframe(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """predict() should return DataFrame with date and prediction columns."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        ensemble = EnsembleModel(models=[naive, ma])
        ensemble.fit(sample_train_df, target="daily_logins")
        predictions = ensemble.predict(horizon=7)

        assert isinstance(predictions, pd.DataFrame)
        assert "date" in predictions.columns
        assert "prediction" in predictions.columns
        assert len(predictions) == 7

    def test_ensemble_weighted_average(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """Predictions should be weighted average of component models."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()  # Predicts 260 (last value)
        ma = MovingAverageModel(window=7)  # Predicts 230 (mean of last 7)

        # Weight naive at 0.7, ma at 0.3
        weights = [0.7, 0.3]
        ensemble = EnsembleModel(models=[naive, ma], weights=weights)
        ensemble.fit(sample_train_df, target="daily_logins")
        predictions = ensemble.predict(horizon=3)

        # Expected: 0.7 * 260 + 0.3 * 230 = 182 + 69 = 251
        expected_prediction = 0.7 * 260 + 0.3 * 230
        assert all(predictions["prediction"] == expected_prediction)

    def test_ensemble_equal_weights(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """With equal weights (None), should be simple average."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()  # Predicts 260
        ma = MovingAverageModel(window=7)  # Predicts 230

        # No weights means equal weights (0.5 each)
        ensemble = EnsembleModel(models=[naive, ma])
        ensemble.fit(sample_train_df, target="daily_logins")
        predictions = ensemble.predict(horizon=3)

        # Expected: (260 + 230) / 2 = 245
        expected_prediction = (260 + 230) / 2
        assert all(predictions["prediction"] == expected_prediction)

    def test_ensemble_default_name(self) -> None:
        """EnsembleModel should have default name 'ensemble'."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        ensemble = EnsembleModel(models=[naive, ma])
        assert ensemble.name == "ensemble"

    def test_ensemble_custom_name(self) -> None:
        """EnsembleModel should accept custom name."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        ensemble = EnsembleModel(models=[naive, ma], name="my_ensemble")
        assert ensemble.name == "my_ensemble"

    def test_ensemble_get_params_returns_dict(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """get_params() should return a dict with model names and weights."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        weights = [0.6, 0.4]
        ensemble = EnsembleModel(models=[naive, ma], weights=weights)
        params = ensemble.get_params()

        assert isinstance(params, dict)
        assert "name" in params
        assert params["name"] == "ensemble"
        assert "model_names" in params
        assert params["model_names"] == ["naive", "moving_average"]
        assert "weights" in params
        assert params["weights"] == [0.6, 0.4]

    def test_ensemble_raises_before_fit(self) -> None:
        """predict() should raise if model hasn't been fit."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        ensemble = EnsembleModel(models=[naive, ma])
        with pytest.raises(ValueError, match="fit"):
            ensemble.predict(horizon=7)

    def test_ensemble_prediction_dates_are_sequential(
        self, sample_train_df: pd.DataFrame
    ) -> None:
        """Prediction dates should start after training data and be sequential."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        ensemble = EnsembleModel(models=[naive, ma])
        ensemble.fit(sample_train_df, target="daily_logins")
        predictions = ensemble.predict(horizon=7)

        # First prediction should be day after last training date (2024-01-15)
        last_train_date = sample_train_df["date"].iloc[-1]
        expected_start = pd.Timestamp(last_train_date) + pd.Timedelta(days=1)
        assert pd.Timestamp(predictions["date"].iloc[0]) == expected_start

        # Dates should be sequential
        expected_dates = pd.date_range(start=expected_start, periods=7, freq="D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(predictions["date"]),
            expected_dates,
            check_names=False,
        )

    def test_ensemble_weights_must_sum_to_one(self) -> None:
        """Weights should sum to 1.0."""
        from volume_forecast.models.baselines import MovingAverageModel, NaiveModel
        from volume_forecast.models.ensemble import EnsembleModel

        naive = NaiveModel()
        ma = MovingAverageModel()
        # Weights that don't sum to 1.0 should raise
        with pytest.raises(ValueError, match="sum"):
            EnsembleModel(models=[naive, ma], weights=[0.5, 0.3])


class TestModelRegistry:
    """Test suite for model registry."""

    def test_registry_get_returns_model(self) -> None:
        """get() should return instantiated model."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.registry import ModelRegistry

        model = ModelRegistry.get("naive")
        assert isinstance(model, BaseModel)
        assert model.name == "naive"

    def test_registry_list_available(self) -> None:
        """list_available() should return list of model names."""
        from volume_forecast.models.registry import ModelRegistry

        available = ModelRegistry.list_available()
        assert isinstance(available, list)
        assert "naive" in available
        assert "seasonal_naive" in available
        assert "moving_average" in available
        assert "arima" in available
        assert "prophet" in available
        assert "lightgbm" in available
        assert "xgboost" in available

    def test_registry_unknown_model_raises(self) -> None:
        """get() with unknown name should raise ValueError."""
        from volume_forecast.models.registry import ModelRegistry

        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.get("nonexistent_model")

    def test_registry_get_with_params(self) -> None:
        """get() should pass kwargs to model constructor."""
        from volume_forecast.models.registry import ModelRegistry

        # Test with MovingAverageModel which accepts window parameter
        model = ModelRegistry.get("moving_average", window=14)
        params = model.get_params()
        assert params["window"] == 14

        # Test with SeasonalNaiveModel which accepts season_length parameter
        model = ModelRegistry.get("seasonal_naive", season_length=14)
        params = model.get_params()
        assert params["season_length"] == 14

    def test_registry_register_new_model(self) -> None:
        """register() should add new model to registry."""
        from volume_forecast.models.base import BaseModel
        from volume_forecast.models.registry import ModelRegistry

        # Create a dummy model class for testing
        class DummyModel(BaseModel):
            def __init__(self, name: str = "dummy") -> None:
                super().__init__(name)

            def fit(self, train_df, target: str):
                return self

            def predict(self, df, horizon: int):
                import pandas as pd
                return pd.DataFrame({"date": [], "prediction": []})

        # Register it
        ModelRegistry.register("dummy_test", DummyModel)

        # Verify it's in the list
        assert "dummy_test" in ModelRegistry.list_available()

        # Verify we can instantiate it
        model = ModelRegistry.get("dummy_test")
        assert isinstance(model, DummyModel)

        # Clean up - remove from registry after test
        del ModelRegistry.MODELS["dummy_test"]

    def test_registry_models_attribute_is_dict(self) -> None:
        """MODELS class attribute should be a dict mapping names to classes."""
        from volume_forecast.models.registry import ModelRegistry

        assert hasattr(ModelRegistry, "MODELS")
        assert isinstance(ModelRegistry.MODELS, dict)
        # Should have at least the 7 required models
        assert len(ModelRegistry.MODELS) >= 7
