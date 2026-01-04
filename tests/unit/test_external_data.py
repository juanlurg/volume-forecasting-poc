"""Tests for external data module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestBaseAPIClient:
    """Test suite for base API client."""

    def test_cache_path_created(self, tmp_path: Path) -> None:
        """Cache directory should be created if it doesn't exist."""
        from volume_forecast.external_data.base import BaseAPIClient

        cache_dir = tmp_path / "cache"
        client = BaseAPIClient(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_get_cached_returns_none_when_no_cache(self, tmp_path: Path) -> None:
        """get_cached should return None when no cache exists."""
        from volume_forecast.external_data.base import BaseAPIClient

        client = BaseAPIClient(cache_dir=tmp_path)
        result = client.get_cached("nonexistent_key")
        assert result is None

    def test_save_and_get_cached(self, tmp_path: Path) -> None:
        """Should be able to save and retrieve cached data."""
        from volume_forecast.external_data.base import BaseAPIClient

        client = BaseAPIClient(cache_dir=tmp_path)
        test_data = {"key": "value", "numbers": [1, 2, 3]}

        client.save_to_cache("test_key", test_data)
        result = client.get_cached("test_key")

        assert result == test_data
