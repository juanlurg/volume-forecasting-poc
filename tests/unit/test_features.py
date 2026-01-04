"""Tests for feature engineering module."""

from datetime import date

import pandas as pd
import pytest


class TestBaseTransformer:
    """Test suite for base transformer."""

    def test_fit_returns_self(self) -> None:
        """fit() should return self for chaining."""
        from volume_forecast.features.base import BaseTransformer

        class DummyTransformer(BaseTransformer):
            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

        transformer = DummyTransformer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = transformer.fit(df)

        assert result is transformer

    def test_fit_transform_chains(self) -> None:
        """fit_transform() should work correctly."""
        from volume_forecast.features.base import BaseTransformer

        class AddOneTransformer(BaseTransformer):
            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                return df.assign(b=df["a"] + 1)

        transformer = AddOneTransformer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = transformer.fit_transform(df)

        assert "b" in result.columns
        assert list(result["b"]) == [2, 3, 4]
