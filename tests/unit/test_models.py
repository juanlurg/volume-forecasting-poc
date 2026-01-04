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
