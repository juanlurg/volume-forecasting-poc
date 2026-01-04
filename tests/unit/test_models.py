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
