"""Static event calendars for racing, tennis, and boxing."""

from datetime import date
from typing import Any


class StaticEventsCalendar:
    """Calendar of major sporting events with static dates.

    These events have predictable dates each year and don't require API calls.
    """

    def __init__(self) -> None:
        """Initialize the calendar."""
        self._racing_events: dict[int, list[dict[str, Any]]] = {}
        self._tennis_events: dict[int, list[dict[str, Any]]] = {}
        self._boxing_events: dict[int, list[dict[str, Any]]] = {}

    def _generate_racing_events(self, year: int) -> list[dict[str, Any]]:
        """Generate racing events for a year.

        Major UK horse racing events with approximate dates.
        Actual dates vary slightly each year.
        """
        return [
            # Cheltenham Festival (mid-March, 4 days)
            {
                "date": date(year, 3, 12),
                "name": "Cheltenham Festival Day 1",
                "event_type": "racing",
                "importance": "high",
            },
            {
                "date": date(year, 3, 13),
                "name": "Cheltenham Festival Day 2",
                "event_type": "racing",
                "importance": "high",
            },
            {
                "date": date(year, 3, 14),
                "name": "Cheltenham Festival Day 3",
                "event_type": "racing",
                "importance": "high",
            },
            {
                "date": date(year, 3, 15),
                "name": "Cheltenham Festival Gold Cup",
                "event_type": "racing",
                "importance": "major",
            },
            # Grand National (early April)
            {
                "date": date(year, 4, 13),
                "name": "Grand National",
                "event_type": "racing",
                "importance": "major",
            },
            # Royal Ascot (mid-June, 5 days)
            {
                "date": date(year, 6, 18),
                "name": "Royal Ascot Day 1",
                "event_type": "racing",
                "importance": "high",
            },
            {
                "date": date(year, 6, 19),
                "name": "Royal Ascot Day 2",
                "event_type": "racing",
                "importance": "high",
            },
            {
                "date": date(year, 6, 20),
                "name": "Royal Ascot Gold Cup",
                "event_type": "racing",
                "importance": "major",
            },
            {
                "date": date(year, 6, 21),
                "name": "Royal Ascot Day 4",
                "event_type": "racing",
                "importance": "high",
            },
            {
                "date": date(year, 6, 22),
                "name": "Royal Ascot Day 5",
                "event_type": "racing",
                "importance": "high",
            },
            # King George VI Chase (Boxing Day)
            {
                "date": date(year, 12, 26),
                "name": "King George VI Chase",
                "event_type": "racing",
                "importance": "high",
            },
        ]

    def _generate_tennis_events(self, year: int) -> list[dict[str, Any]]:
        """Generate tennis Grand Slam events for a year."""
        return [
            # Australian Open (mid-January, 2 weeks)
            {
                "date": date(year, 1, 14),
                "name": "Australian Open Start",
                "event_type": "tennis",
                "importance": "medium",
            },
            {
                "date": date(year, 1, 28),
                "name": "Australian Open Final",
                "event_type": "tennis",
                "importance": "high",
            },
            # French Open (late May - early June)
            {
                "date": date(year, 5, 26),
                "name": "French Open Start",
                "event_type": "tennis",
                "importance": "medium",
            },
            {
                "date": date(year, 6, 9),
                "name": "French Open Final",
                "event_type": "tennis",
                "importance": "high",
            },
            # Wimbledon (late June - mid July)
            {
                "date": date(year, 7, 1),
                "name": "Wimbledon Start",
                "event_type": "tennis",
                "importance": "high",
            },
            {
                "date": date(year, 7, 14),
                "name": "Wimbledon Final",
                "event_type": "tennis",
                "importance": "major",
            },
            # US Open (late August - early September)
            {
                "date": date(year, 8, 26),
                "name": "US Open Start",
                "event_type": "tennis",
                "importance": "medium",
            },
            {
                "date": date(year, 9, 8),
                "name": "US Open Final",
                "event_type": "tennis",
                "importance": "high",
            },
        ]

    def _generate_boxing_events(self, year: int) -> list[dict[str, Any]]:
        """Generate major boxing/UFC events.

        Note: These are placeholder dates. Real events vary significantly.
        """
        return [
            # Typical major boxing events (Saturday nights)
            {
                "date": date(year, 3, 9),
                "name": "Major Boxing Event (Spring)",
                "event_type": "boxing",
                "importance": "high",
            },
            {
                "date": date(year, 6, 1),
                "name": "Major Boxing Event (Summer)",
                "event_type": "boxing",
                "importance": "high",
            },
            {
                "date": date(year, 9, 21),
                "name": "Major Boxing Event (Autumn)",
                "event_type": "boxing",
                "importance": "high",
            },
            {
                "date": date(year, 12, 21),
                "name": "Major Boxing Event (Winter)",
                "event_type": "boxing",
                "importance": "high",
            },
        ]

    def get_racing_events(self, year: int) -> list[dict[str, Any]]:
        """Get horse racing events for a year.

        Args:
            year: Year to get events for.

        Returns:
            List of racing events.
        """
        if year not in self._racing_events:
            self._racing_events[year] = self._generate_racing_events(year)
        return self._racing_events[year]

    def get_tennis_events(self, year: int) -> list[dict[str, Any]]:
        """Get tennis events for a year.

        Args:
            year: Year to get events for.

        Returns:
            List of tennis events.
        """
        if year not in self._tennis_events:
            self._tennis_events[year] = self._generate_tennis_events(year)
        return self._tennis_events[year]

    def get_boxing_events(self, year: int) -> list[dict[str, Any]]:
        """Get boxing/UFC events for a year.

        Args:
            year: Year to get events for.

        Returns:
            List of boxing events.
        """
        if year not in self._boxing_events:
            self._boxing_events[year] = self._generate_boxing_events(year)
        return self._boxing_events[year]

    def get_all_events(self, year: int) -> list[dict[str, Any]]:
        """Get all static events for a year.

        Args:
            year: Year to get events for.

        Returns:
            List of all events sorted by date.
        """
        events = (
            self.get_racing_events(year)
            + self.get_tennis_events(year)
            + self.get_boxing_events(year)
        )
        return sorted(events, key=lambda x: x["date"])

    def get_events_in_range(
        self, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """Get all events in a date range.

        Args:
            start_date: Start of range.
            end_date: End of range.

        Returns:
            List of events in range.
        """
        events = []
        for year in range(start_date.year, end_date.year + 1):
            for event in self.get_all_events(year):
                if start_date <= event["date"] <= end_date:
                    events.append(event)
        return sorted(events, key=lambda x: x["date"])
