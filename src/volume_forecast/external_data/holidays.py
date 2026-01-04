"""UK Bank Holidays API client."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

from volume_forecast.external_data.base import BaseAPIClient


class UKHolidaysClient(BaseAPIClient):
    """Client for UK Government Bank Holidays API."""

    API_URL = "https://www.gov.uk/bank-holidays.json"

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the holidays client."""
        super().__init__(cache_dir=cache_dir)
        self._holidays: list[dict[str, Any]] | None = None

    def _parse_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse the API response into a list of holiday events.

        Args:
            response: Raw API response.

        Returns:
            List of holiday dictionaries.
        """
        holidays = []
        # Use England and Wales holidays
        events = response.get("england-and-wales", {}).get("events", [])

        for event in events:
            holiday_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
            holidays.append({
                "date": holiday_date,
                "name": event["title"],
                "event_type": "holiday",
                "importance": self._get_importance(event["title"]),
            })

        return sorted(holidays, key=lambda x: x["date"])

    def _get_importance(self, name: str) -> str:
        """Determine importance level based on holiday name.

        Args:
            name: Holiday name.

        Returns:
            Importance level (low, medium, high, major).
        """
        major_holidays = ["christmas", "boxing day", "new year"]
        high_holidays = ["good friday", "easter"]

        name_lower = name.lower()
        for major in major_holidays:
            if major in name_lower:
                return "major"
        for high in high_holidays:
            if high in name_lower:
                return "high"
        return "medium"

    def fetch_holidays(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """Fetch UK bank holidays.

        Args:
            use_cache: Whether to use cached data.

        Returns:
            List of holiday dictionaries.
        """
        cache_key = "uk_bank_holidays"

        if use_cache:
            cached = self.get_cached(cache_key)
            if cached is not None:
                # Convert date strings back to date objects
                for h in cached:
                    if isinstance(h["date"], str):
                        h["date"] = datetime.strptime(h["date"], "%Y-%m-%d").date()
                self._holidays = cached
                return cached

        try:
            response = self.fetch(self.API_URL)
            holidays = self._parse_response(response)
            # Save with date as string for JSON serialization
            cache_data = [
                {**h, "date": h["date"].isoformat()} for h in holidays
            ]
            self.save_to_cache(cache_key, cache_data)
            self._holidays = holidays
            return holidays
        except Exception:
            # Fall back to cache if available
            cached = self.get_cached(cache_key)
            if cached is not None:
                for h in cached:
                    if isinstance(h["date"], str):
                        h["date"] = datetime.strptime(h["date"], "%Y-%m-%d").date()
                self._holidays = cached
                return cached
            raise

    def is_holiday(self, check_date: date) -> bool:
        """Check if a date is a UK bank holiday.

        Args:
            check_date: Date to check.

        Returns:
            True if the date is a bank holiday.
        """
        if self._holidays is None:
            self.fetch_holidays()
        return any(h["date"] == check_date for h in self._holidays or [])

    def get_holiday(self, check_date: date) -> dict[str, Any] | None:
        """Get holiday info for a date.

        Args:
            check_date: Date to check.

        Returns:
            Holiday dictionary or None.
        """
        if self._holidays is None:
            self.fetch_holidays()
        for h in self._holidays or []:
            if h["date"] == check_date:
                return h
        return None

    def get_holidays_in_range(
        self, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """Get all holidays in a date range.

        Args:
            start_date: Start of range.
            end_date: End of range.

        Returns:
            List of holidays in the range.
        """
        if self._holidays is None:
            self.fetch_holidays()
        return [
            h for h in self._holidays or []
            if start_date <= h["date"] <= end_date
        ]
