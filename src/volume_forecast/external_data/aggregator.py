"""Event aggregator combining all event sources."""

from datetime import date
from pathlib import Path
from typing import Any

from volume_forecast.external_data.football import FootballClient
from volume_forecast.external_data.holidays import UKHolidaysClient
from volume_forecast.external_data.static_events import StaticEventsCalendar


class EventAggregator:
    """Aggregates events from all sources into a unified calendar."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        football_api_key: str = "",
    ) -> None:
        """Initialize the aggregator.

        Args:
            cache_dir: Cache directory for API responses.
            football_api_key: API key for football-data.org.
        """
        self.cache_dir = cache_dir or Path("data/external")
        self.holidays_client = UKHolidaysClient(cache_dir=self.cache_dir)
        self.football_client = FootballClient(
            api_key=football_api_key, cache_dir=self.cache_dir
        )
        self.static_calendar = StaticEventsCalendar()

    def get_events(
        self,
        start_date: date,
        end_date: date,
        include_holidays: bool = True,
        include_football: bool = True,
        include_racing: bool = True,
        include_tennis: bool = True,
        include_boxing: bool = True,
    ) -> list[dict[str, Any]]:
        """Get all events in a date range.

        Args:
            start_date: Start of range.
            end_date: End of range.
            include_holidays: Include UK bank holidays.
            include_football: Include football matches (requires API).
            include_racing: Include horse racing events.
            include_tennis: Include tennis Grand Slams.
            include_boxing: Include boxing/UFC events.

        Returns:
            List of events sorted by date.
        """
        events: list[dict[str, Any]] = []

        # UK Bank Holidays
        if include_holidays:
            try:
                holidays = self.holidays_client.get_holidays_in_range(
                    start_date, end_date
                )
                events.extend(holidays)
            except Exception:
                pass  # Continue without holidays if fetch fails

        # Football matches
        if include_football:
            try:
                pl_matches = self.football_client.get_premier_league_matches(
                    start_date, end_date
                )
                events.extend(pl_matches)
            except Exception:
                pass  # Continue without football if fetch fails

        # Static events
        static_events = self.static_calendar.get_events_in_range(
            start_date, end_date
        )
        for event in static_events:
            if event["event_type"] == "racing" and include_racing:
                events.append(event)
            elif event["event_type"] == "tennis" and include_tennis:
                events.append(event)
            elif event["event_type"] == "boxing" and include_boxing:
                events.append(event)

        # Sort by date and deduplicate
        events = sorted(events, key=lambda x: (x["date"], x["name"]))
        return events

    def get_events_for_date(self, target_date: date) -> list[dict[str, Any]]:
        """Get all events for a specific date.

        Args:
            target_date: Date to get events for.

        Returns:
            List of events on that date.
        """
        return self.get_events(target_date, target_date)

    def has_major_event(self, target_date: date) -> bool:
        """Check if a date has a major event.

        Args:
            target_date: Date to check.

        Returns:
            True if there's a major event.
        """
        events = self.get_events_for_date(target_date)
        return any(e.get("importance") == "major" for e in events)

    def get_max_importance(self, target_date: date) -> str:
        """Get the maximum event importance for a date.

        Args:
            target_date: Date to check.

        Returns:
            Maximum importance level or "none".
        """
        events = self.get_events_for_date(target_date)
        if not events:
            return "none"

        importance_order = ["low", "medium", "high", "major"]
        max_importance = "low"
        for event in events:
            imp = event.get("importance", "low")
            if importance_order.index(imp) > importance_order.index(max_importance):
                max_importance = imp
        return max_importance
