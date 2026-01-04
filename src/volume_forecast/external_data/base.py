"""Base API client with caching support."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx


class BaseAPIClient:
    """Base class for API clients with caching and retry logic."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 24,
        timeout: float = 30.0,
        retries: int = 3,
    ) -> None:
        """Initialize the API client.

        Args:
            cache_dir: Directory for caching responses.
            cache_ttl_hours: Cache time-to-live in hours.
            timeout: Request timeout in seconds.
            retries: Number of retry attempts.
        """
        self.cache_dir = cache_dir or Path("data/external")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.timeout = timeout
        self.retries = retries
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-loaded HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    def get_cached(self, key: str) -> dict[str, Any] | None:
        """Get cached data if it exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            Cached data or None if not found/expired.
        """
        cache_path = self._cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                cached = json.load(f)

            cached_at = datetime.fromisoformat(cached["_cached_at"])
            if datetime.now() - cached_at > self.cache_ttl:
                return None

            return cached["data"]
        except (json.JSONDecodeError, KeyError):
            return None

    def save_to_cache(self, key: str, data: dict[str, Any]) -> None:
        """Save data to cache.

        Args:
            key: Cache key.
            data: Data to cache.
        """
        cache_path = self._cache_path(key)
        cached = {
            "_cached_at": datetime.now().isoformat(),
            "data": data,
        }
        with open(cache_path, "w") as f:
            json.dump(cached, f, indent=2, default=str)

    def fetch(self, url: str, params: dict | None = None) -> dict[str, Any]:
        """Fetch data from URL with retry logic.

        Args:
            url: URL to fetch.
            params: Query parameters.

        Returns:
            Response JSON data.

        Raises:
            httpx.HTTPError: If all retries fail.
        """
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self.retries - 1:
                    continue
        raise last_error  # type: ignore

    def fetch_with_cache(
        self, url: str, cache_key: str, params: dict | None = None
    ) -> dict[str, Any]:
        """Fetch data with caching.

        Args:
            url: URL to fetch.
            cache_key: Key for caching.
            params: Query parameters.

        Returns:
            Response data (from cache or fresh).
        """
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        data = self.fetch(url, params)
        self.save_to_cache(cache_key, data)
        return data

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "BaseAPIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
