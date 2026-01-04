"""Football events API client using football-data.org."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

from volume_forecast.external_data.base import BaseAPIClient


class FootballClient(BaseAPIClient):
    """Client for football-data.org API."""

    API_URL = "https://api.football-data.org/v4"

    # Premier League "Big Six" teams (normalized names for matching)
    BIG_SIX = {
        "Arsenal",
        "Chelsea",
        "Liverpool",
        "Manchester City",
        "Manchester United",
        "Tottenham Hotspur",
        "Tottenham",
        "Spurs",
    }

    # Competition IDs
    COMPETITIONS = {
        "PL": "Premier League",
        "CL": "UEFA Champions League",
        "EC": "European Championship",
        "WC": "FIFA World Cup",
        "FA": "FA Cup",
    }

    def __init__(
        self,
        api_key: str = "",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the football client.

        Args:
            api_key: football-data.org API key.
            cache_dir: Cache directory.
        """
        super().__init__(cache_dir=cache_dir)
        self.api_key = api_key

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Auth-Token"] = self.api_key
        return headers

    def _determine_importance(
        self,
        competition: str,
        stage: str,
        home_team: str,
        away_team: str,
    ) -> str:
        """Determine event importance based on match details.

        Args:
            competition: Competition name.
            stage: Match stage (FINAL, SEMI_FINAL, etc.).
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            Importance level.
        """
        # Finals and semi-finals of major competitions
        if stage in ("FINAL", "SEMI_FINALS"):
            if "Champions League" in competition or "World Cup" in competition:
                return "major"
            if "European Championship" in competition:
                return "major"
            return "high"

        # Quarter-finals of major competitions
        if stage == "QUARTER_FINALS":
            if "Champions League" in competition or "World Cup" in competition:
                return "high"
            return "medium"

        # Big Six matches in Premier League
        if "Premier League" in competition:
            home_is_big_six = any(team in home_team for team in self.BIG_SIX)
            away_is_big_six = any(team in away_team for team in self.BIG_SIX)
            if home_is_big_six and away_is_big_six:
                return "high"

        # Regular Premier League matches
        if "Premier League" in competition:
            return "medium"

        # Other matches
        return "low"

    def fetch_matches(
        self,
        competition: str,
        start_date: date,
        end_date: date,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch matches for a competition in a date range.

        Args:
            competition: Competition code (PL, CL, etc.).
            start_date: Start date.
            end_date: End date.
            use_cache: Whether to use cache.

        Returns:
            List of match dictionaries.
        """
        cache_key = f"football_{competition}_{start_date}_{end_date}"

        if use_cache:
            cached = self.get_cached(cache_key)
            if cached is not None:
                # Convert date strings
                for m in cached:
                    if isinstance(m["date"], str):
                        m["date"] = datetime.strptime(m["date"], "%Y-%m-%d").date()
                return cached

        url = f"{self.API_URL}/competitions/{competition}/matches"
        params = {
            "dateFrom": start_date.isoformat(),
            "dateTo": end_date.isoformat(),
        }

        try:
            # Note: Without API key, requests are rate-limited
            response = self.client.get(
                url,
                params=params,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            matches = self._parse_matches(data, competition)

            # Cache with string dates
            cache_data = [{**m, "date": m["date"].isoformat()} for m in matches]
            self.save_to_cache(cache_key, cache_data)

            return matches
        except Exception:
            # Return cached data if available
            cached = self.get_cached(cache_key)
            if cached:
                for m in cached:
                    if isinstance(m["date"], str):
                        m["date"] = datetime.strptime(m["date"], "%Y-%m-%d").date()
                return cached
            return []

    def _parse_matches(
        self, response: dict[str, Any], competition: str
    ) -> list[dict[str, Any]]:
        """Parse API response into match events.

        Args:
            response: Raw API response.
            competition: Competition code.

        Returns:
            List of match dictionaries.
        """
        matches = []
        competition_name = self.COMPETITIONS.get(competition, competition)

        for match in response.get("matches", []):
            match_date = datetime.fromisoformat(
                match["utcDate"].replace("Z", "+00:00")
            ).date()
            home_team = match.get("homeTeam", {}).get("name", "Unknown")
            away_team = match.get("awayTeam", {}).get("name", "Unknown")
            stage = match.get("stage", "REGULAR_SEASON")

            importance = self._determine_importance(
                competition_name, stage, home_team, away_team
            )

            matches.append({
                "date": match_date,
                "event_type": "football",
                "name": f"{home_team} vs {away_team}",
                "competition": competition_name,
                "importance": importance,
                "stage": stage,
            })

        return sorted(matches, key=lambda x: x["date"])

    def get_premier_league_matches(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """Get Premier League matches in date range.

        Args:
            start_date: Start date.
            end_date: End date.

        Returns:
            List of matches.
        """
        return self.fetch_matches("PL", start_date, end_date)
