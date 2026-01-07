"""Static event calendars for racing, tennis, and boxing.

Contains exact dates for major sporting events for 2023, 2024, and 2025.
Sources:
- Cheltenham Festival: https://www.cheltenhamfestival.org.uk/
- Grand National: https://en.wikipedia.org/wiki/Grand_National
- Royal Ascot: https://www.ascot.com/
- Tennis Grand Slams: ATP Tour, Wikipedia
- Boxing: Sky Sports, Wikipedia
"""

from datetime import date
from typing import Any


# =============================================================================
# HORSE RACING EVENTS - Exact dates by year
# =============================================================================

CHELTENHAM_FESTIVAL_DATES: dict[int, tuple[date, date, date, date]] = {
    # (Day 1, Day 2, Day 3, Gold Cup Day)
    2023: (date(2023, 3, 14), date(2023, 3, 15), date(2023, 3, 16), date(2023, 3, 17)),
    2024: (date(2024, 3, 12), date(2024, 3, 13), date(2024, 3, 14), date(2024, 3, 15)),
    2025: (date(2025, 3, 11), date(2025, 3, 12), date(2025, 3, 13), date(2025, 3, 14)),
}

GRAND_NATIONAL_DATES: dict[int, date] = {
    2023: date(2023, 4, 15),  # Saturday
    2024: date(2024, 4, 13),  # Saturday
    2025: date(2025, 4, 5),   # Saturday
}

ROYAL_ASCOT_DATES: dict[int, tuple[date, date, date, date, date]] = {
    # (Day 1, Day 2, Gold Cup Day, Day 4, Day 5)
    2023: (
        date(2023, 6, 20), date(2023, 6, 21), date(2023, 6, 22),
        date(2023, 6, 23), date(2023, 6, 24)
    ),
    2024: (
        date(2024, 6, 18), date(2024, 6, 19), date(2024, 6, 20),
        date(2024, 6, 21), date(2024, 6, 22)
    ),
    2025: (
        date(2025, 6, 17), date(2025, 6, 18), date(2025, 6, 19),
        date(2025, 6, 20), date(2025, 6, 21)
    ),
}

# King George VI Chase is always on Boxing Day (December 26)
KING_GEORGE_DATES: dict[int, date] = {
    2023: date(2023, 12, 26),
    2024: date(2024, 12, 26),
    2025: date(2025, 12, 26),
}

# =============================================================================
# TENNIS GRAND SLAM EVENTS - Exact dates by year
# =============================================================================

AUSTRALIAN_OPEN_DATES: dict[int, tuple[date, date]] = {
    # (Start date, Final date)
    2023: (date(2023, 1, 16), date(2023, 1, 29)),
    2024: (date(2024, 1, 14), date(2024, 1, 28)),
    2025: (date(2025, 1, 12), date(2025, 1, 26)),
}

FRENCH_OPEN_DATES: dict[int, tuple[date, date]] = {
    # (Start date, Final date)
    2023: (date(2023, 5, 28), date(2023, 6, 11)),
    2024: (date(2024, 5, 26), date(2024, 6, 9)),
    2025: (date(2025, 5, 25), date(2025, 6, 8)),
}

WIMBLEDON_DATES: dict[int, tuple[date, date]] = {
    # (Start date, Final date)
    2023: (date(2023, 7, 3), date(2023, 7, 16)),
    2024: (date(2024, 7, 1), date(2024, 7, 14)),
    2025: (date(2025, 6, 30), date(2025, 7, 13)),
}

US_OPEN_DATES: dict[int, tuple[date, date]] = {
    # (Start date, Final date)
    2023: (date(2023, 8, 28), date(2023, 9, 10)),
    2024: (date(2024, 8, 26), date(2024, 9, 8)),
    2025: (date(2025, 8, 24), date(2025, 9, 7)),
}

# =============================================================================
# MAJOR BOXING EVENTS - Exact dates by year (UK-relevant heavyweight fights)
# =============================================================================

BOXING_EVENTS: dict[int, list[tuple[date, str, str]]] = {
    # (date, name, importance)
    2023: [
        (date(2023, 4, 1), "Anthony Joshua vs Jermaine Franklin", "high"),
        (date(2023, 8, 12), "Anthony Joshua vs Robert Helenius", "high"),
        (date(2023, 8, 26), "Usyk vs Dubois", "major"),
        (date(2023, 10, 28), "Tyson Fury vs Francis Ngannou", "major"),
        (date(2023, 12, 23), "Anthony Joshua vs Otto Wallin", "high"),
    ],
    2024: [
        (date(2024, 3, 8), "Anthony Joshua vs Francis Ngannou", "major"),
        (date(2024, 5, 18), "Fury vs Usyk I - Undisputed", "major"),
        (date(2024, 9, 21), "Anthony Joshua vs Daniel Dubois", "major"),
        (date(2024, 12, 21), "Fury vs Usyk II - Rematch", "major"),
    ],
    2025: [
        (date(2025, 2, 22), "Chris Eubank Jr vs Conor Benn", "high"),
        (date(2025, 7, 19), "Usyk vs Dubois 2 - Wembley", "major"),
    ],
}


class StaticEventsCalendar:
    """Calendar of major sporting events with exact dates for 2023-2025.

    These events have been researched and verified with exact dates.
    """

    def __init__(self) -> None:
        """Initialize the calendar."""
        self._racing_events: dict[int, list[dict[str, Any]]] = {}
        self._tennis_events: dict[int, list[dict[str, Any]]] = {}
        self._boxing_events: dict[int, list[dict[str, Any]]] = {}

    def _generate_racing_events(self, year: int) -> list[dict[str, Any]]:
        """Generate racing events for a year.

        Uses exact dates for 2023-2025, falls back to approximations for other years.
        """
        events = []

        # Cheltenham Festival
        if year in CHELTENHAM_FESTIVAL_DATES:
            days = CHELTENHAM_FESTIVAL_DATES[year]
            events.extend([
                {"date": days[0], "name": "Cheltenham Festival Day 1",
                 "event_type": "racing", "importance": "high"},
                {"date": days[1], "name": "Cheltenham Festival Day 2",
                 "event_type": "racing", "importance": "high"},
                {"date": days[2], "name": "Cheltenham Festival Day 3",
                 "event_type": "racing", "importance": "high"},
                {"date": days[3], "name": "Cheltenham Festival Gold Cup",
                 "event_type": "racing", "importance": "major"},
            ])
        else:
            # Fallback: typically mid-March, Tuesday-Friday
            events.extend([
                {"date": date(year, 3, 12), "name": "Cheltenham Festival Day 1",
                 "event_type": "racing", "importance": "high"},
                {"date": date(year, 3, 13), "name": "Cheltenham Festival Day 2",
                 "event_type": "racing", "importance": "high"},
                {"date": date(year, 3, 14), "name": "Cheltenham Festival Day 3",
                 "event_type": "racing", "importance": "high"},
                {"date": date(year, 3, 15), "name": "Cheltenham Festival Gold Cup",
                 "event_type": "racing", "importance": "major"},
            ])

        # Grand National
        if year in GRAND_NATIONAL_DATES:
            events.append({
                "date": GRAND_NATIONAL_DATES[year],
                "name": "Grand National",
                "event_type": "racing",
                "importance": "major",
            })
        else:
            # Fallback: first or second Saturday in April
            events.append({
                "date": date(year, 4, 6),
                "name": "Grand National",
                "event_type": "racing",
                "importance": "major",
            })

        # Royal Ascot
        if year in ROYAL_ASCOT_DATES:
            days = ROYAL_ASCOT_DATES[year]
            events.extend([
                {"date": days[0], "name": "Royal Ascot Day 1",
                 "event_type": "racing", "importance": "high"},
                {"date": days[1], "name": "Royal Ascot Day 2",
                 "event_type": "racing", "importance": "high"},
                {"date": days[2], "name": "Royal Ascot Gold Cup",
                 "event_type": "racing", "importance": "major"},
                {"date": days[3], "name": "Royal Ascot Day 4",
                 "event_type": "racing", "importance": "high"},
                {"date": days[4], "name": "Royal Ascot Day 5",
                 "event_type": "racing", "importance": "high"},
            ])
        else:
            # Fallback: typically third week of June, Tuesday-Saturday
            events.extend([
                {"date": date(year, 6, 18), "name": "Royal Ascot Day 1",
                 "event_type": "racing", "importance": "high"},
                {"date": date(year, 6, 19), "name": "Royal Ascot Day 2",
                 "event_type": "racing", "importance": "high"},
                {"date": date(year, 6, 20), "name": "Royal Ascot Gold Cup",
                 "event_type": "racing", "importance": "major"},
                {"date": date(year, 6, 21), "name": "Royal Ascot Day 4",
                 "event_type": "racing", "importance": "high"},
                {"date": date(year, 6, 22), "name": "Royal Ascot Day 5",
                 "event_type": "racing", "importance": "high"},
            ])

        # King George VI Chase - always Boxing Day
        events.append({
            "date": date(year, 12, 26),
            "name": "King George VI Chase",
            "event_type": "racing",
            "importance": "high",
        })

        return events

    def _generate_tennis_events(self, year: int) -> list[dict[str, Any]]:
        """Generate tennis Grand Slam events for a year.

        Uses exact dates for 2023-2025, falls back to approximations for other years.
        """
        events = []

        # Australian Open
        if year in AUSTRALIAN_OPEN_DATES:
            start, final = AUSTRALIAN_OPEN_DATES[year]
        else:
            start, final = date(year, 1, 15), date(year, 1, 28)
        events.extend([
            {"date": start, "name": "Australian Open Start",
             "event_type": "tennis", "importance": "medium"},
            {"date": final, "name": "Australian Open Final",
             "event_type": "tennis", "importance": "high"},
        ])

        # French Open
        if year in FRENCH_OPEN_DATES:
            start, final = FRENCH_OPEN_DATES[year]
        else:
            start, final = date(year, 5, 26), date(year, 6, 9)
        events.extend([
            {"date": start, "name": "French Open Start",
             "event_type": "tennis", "importance": "medium"},
            {"date": final, "name": "French Open Final",
             "event_type": "tennis", "importance": "high"},
        ])

        # Wimbledon
        if year in WIMBLEDON_DATES:
            start, final = WIMBLEDON_DATES[year]
        else:
            start, final = date(year, 7, 1), date(year, 7, 14)
        events.extend([
            {"date": start, "name": "Wimbledon Start",
             "event_type": "tennis", "importance": "high"},
            {"date": final, "name": "Wimbledon Final",
             "event_type": "tennis", "importance": "major"},
        ])

        # US Open
        if year in US_OPEN_DATES:
            start, final = US_OPEN_DATES[year]
        else:
            start, final = date(year, 8, 26), date(year, 9, 8)
        events.extend([
            {"date": start, "name": "US Open Start",
             "event_type": "tennis", "importance": "medium"},
            {"date": final, "name": "US Open Final",
             "event_type": "tennis", "importance": "high"},
        ])

        return events

    def _generate_boxing_events(self, year: int) -> list[dict[str, Any]]:
        """Generate major boxing events for a year.

        Uses exact dates for 2023-2025 (UK-relevant heavyweight fights).
        Falls back to placeholder dates for other years.
        """
        if year in BOXING_EVENTS:
            return [
                {
                    "date": event_date,
                    "name": name,
                    "event_type": "boxing",
                    "importance": importance,
                }
                for event_date, name, importance in BOXING_EVENTS[year]
            ]
        else:
            # Fallback: placeholder seasonal events
            return [
                {"date": date(year, 3, 9), "name": "Major Boxing Event (Spring)",
                 "event_type": "boxing", "importance": "high"},
                {"date": date(year, 6, 1), "name": "Major Boxing Event (Summer)",
                 "event_type": "boxing", "importance": "high"},
                {"date": date(year, 9, 21), "name": "Major Boxing Event (Autumn)",
                 "event_type": "boxing", "importance": "high"},
                {"date": date(year, 12, 21), "name": "Major Boxing Event (Winter)",
                 "event_type": "boxing", "importance": "high"},
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
