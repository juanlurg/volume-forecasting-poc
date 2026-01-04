"""Tests for external data module."""

from datetime import date
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


class TestUKHolidaysClient:
    """Test suite for UK holidays client."""

    def test_parse_holiday_response(self) -> None:
        """Should parse UK government API response correctly."""
        from volume_forecast.external_data.holidays import UKHolidaysClient

        mock_response = {
            "england-and-wales": {
                "events": [
                    {"title": "New Year's Day", "date": "2024-01-01"},
                    {"title": "Good Friday", "date": "2024-03-29"},
                ]
            }
        }

        client = UKHolidaysClient()
        holidays = client._parse_response(mock_response)

        assert len(holidays) == 2
        assert holidays[0]["date"] == date(2024, 1, 1)
        assert holidays[0]["name"] == "New Year's Day"
        assert holidays[0]["event_type"] == "holiday"

    def test_is_holiday(self) -> None:
        """Should correctly identify holiday dates."""
        from volume_forecast.external_data.holidays import UKHolidaysClient

        client = UKHolidaysClient()
        # Mock the holidays data
        client._holidays = [
            {"date": date(2024, 1, 1), "name": "New Year's Day"},
            {"date": date(2024, 12, 25), "name": "Christmas Day"},
        ]

        assert client.is_holiday(date(2024, 1, 1)) is True
        assert client.is_holiday(date(2024, 6, 15)) is False
