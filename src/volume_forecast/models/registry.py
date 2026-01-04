"""Model registry for easy model instantiation."""

from typing import Any

from volume_forecast.models.base import BaseModel
from volume_forecast.models.baselines import (
    MovingAverageModel,
    NaiveModel,
    SeasonalNaiveModel,
)
from volume_forecast.models.prophet_model import ProphetModel
from volume_forecast.models.statistical import ARIMAModel
from volume_forecast.models.tree_models import LightGBMModel, XGBoostModel


class ModelRegistry:
    """Registry for forecasting models.

    Provides a central place to register, list, and instantiate models by name.

    Attributes:
        MODELS: Dictionary mapping model names to model classes.

    Example:
        >>> model = ModelRegistry.get("naive")
        >>> model.fit(train_df, target="daily_logins")

        >>> available = ModelRegistry.list_available()
        >>> print(available)
        ['naive', 'seasonal_naive', 'moving_average', 'arima', 'prophet', 'lightgbm', 'xgboost']
    """

    MODELS: dict[str, type[BaseModel]] = {
        "naive": NaiveModel,
        "seasonal_naive": SeasonalNaiveModel,
        "moving_average": MovingAverageModel,
        "arima": ARIMAModel,
        "prophet": ProphetModel,
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
    }

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseModel:
        """Instantiate and return a model by name.

        Args:
            name: Name of the model to instantiate.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            An instantiated model.

        Raises:
            ValueError: If the model name is not in the registry.

        Example:
            >>> model = ModelRegistry.get("moving_average", window=14)
            >>> model.get_params()
            {'name': 'moving_average', 'window': 14}
        """
        if name not in cls.MODELS:
            available = ", ".join(sorted(cls.MODELS.keys()))
            raise ValueError(f"Unknown model: '{name}'. Available models: {available}")

        model_class = cls.MODELS[name]
        return model_class(**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """Return list of available model names.

        Returns:
            List of registered model names.

        Example:
            >>> ModelRegistry.list_available()
            ['arima', 'lightgbm', 'moving_average', 'naive', 'prophet', 'seasonal_naive', 'xgboost']
        """
        return sorted(cls.MODELS.keys())

    @classmethod
    def register(cls, name: str, model_class: type[BaseModel]) -> None:
        """Register a new model class.

        Args:
            name: Name to register the model under.
            model_class: The model class to register.

        Example:
            >>> class MyCustomModel(BaseModel):
            ...     pass
            >>> ModelRegistry.register("custom", MyCustomModel)
            >>> "custom" in ModelRegistry.list_available()
            True
        """
        cls.MODELS[name] = model_class
