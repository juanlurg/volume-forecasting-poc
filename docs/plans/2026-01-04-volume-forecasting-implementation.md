# Volume Forecasting POC - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a POC for daily login and deposit volume forecasting on a UK betting platform with 7-day horizon.

**Architecture:** Modular Python package with src layout. Synthetic data generation simulates 2 years of realistic betting patterns. External APIs provide sports events and UK holidays. Feature engineering transforms data for ML models. Multiple model families (baselines, statistical, ML, ensemble) are benchmarked via walk-forward validation.

**Tech Stack:** Python 3.11+, uv, pandas, numpy, scikit-learn, statsmodels, prophet, xgboost, lightgbm, pygam, pytest, ruff, mypy

---

## Phase 1: Project Setup

### Task 1.1: Initialize Project with uv

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`

**Step 1: Initialize project**

```bash
cd C:\Users\juanlu\dev\volume-forecasting-poc
uv init --lib --name volume_forecast
```

**Step 2: Set Python version**

```bash
echo 3.11 > .python-version
```

**Step 3: Create .gitignore**

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# Virtual environments
.venv/
venv/

# uv
.uv/

# Jupyter
.ipynb_checkpoints/

# Data files
data/raw/*
data/processed/*
data/external/*
!data/*/.gitkeep
*.csv
*.parquet
*.pkl
*.joblib

# Models
models/*
!models/.gitkeep

# IDE
.vscode/*
!.vscode/settings.json
.idea/

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/

# Environment
.env
.env.*
!.env.example

# OS
.DS_Store
Thumbs.db
```

**Step 4: Verify initialization**

```bash
uv sync
```

Expected: Lock file created, basic package structure exists.

**Step 5: Commit**

```bash
git init
git add .
git commit -m "chore: initialize project with uv"
```

---

### Task 1.2: Configure pyproject.toml with Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Replace contents of `pyproject.toml`:

```toml
[project]
name = "volume-forecast"
version = "0.1.0"
description = "Volume forecasting POC for betting platform"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    "statsmodels>=0.14.0",
    "prophet>=1.1.5",
    "xgboost>=2.0.0",
    "lightgbm>=4.3.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
    "httpx>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
]

[project.optional-dependencies]
notebooks = [
    "jupyter>=1.0.0",
    "ipykernel>=6.27.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.9",
    "mypy>=1.8.0",
    "pandas-stubs>=2.1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/volume_forecast"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = ["-ra", "-q", "--strict-markers"]

[tool.ruff]
target-version = "py311"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
ignore_missing_imports = true
```

**Step 2: Sync dependencies**

```bash
uv sync --all-extras
```

Expected: All packages installed successfully.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add project dependencies"
```

---

### Task 1.3: Create Directory Structure

**Files:**
- Create: Multiple directories and placeholder files

**Step 1: Create directories**

```bash
mkdir -p src/volume_forecast/{config,data_generation,external_data,features,models,evaluation}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p data/{raw,processed,external}
mkdir -p models
mkdir -p notebooks
mkdir -p docs/plans
```

**Step 2: Create placeholder files**

```bash
touch src/volume_forecast/__init__.py
touch src/volume_forecast/py.typed
touch src/volume_forecast/config/__init__.py
touch src/volume_forecast/data_generation/__init__.py
touch src/volume_forecast/external_data/__init__.py
touch src/volume_forecast/features/__init__.py
touch src/volume_forecast/models/__init__.py
touch src/volume_forecast/evaluation/__init__.py
touch tests/__init__.py
touch tests/conftest.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep
touch models/.gitkeep
```

**Step 3: Create base __init__.py**

Write to `src/volume_forecast/__init__.py`:

```python
"""Volume Forecasting POC for betting platform."""

__version__ = "0.1.0"
```

**Step 4: Verify package imports**

```bash
uv run python -c "import volume_forecast; print(volume_forecast.__version__)"
```

Expected: `0.1.0`

**Step 5: Commit**

```bash
git add .
git commit -m "chore: create project directory structure"
```

---

## Phase 2: Configuration Module

### Task 2.1: Create Constants

**Files:**
- Create: `src/volume_forecast/config/constants.py`
- Test: `tests/unit/test_config.py`

**Step 1: Write the failing test**

Create `tests/unit/test_config.py`:

```python
"""Tests for configuration module."""

import pytest


class TestConstants:
    """Test suite for constants."""

    def test_day_multipliers_sum_to_seven(self) -> None:
        """Day multipliers should average to ~1.0."""
        from volume_forecast.config.constants import DAY_OF_WEEK_MULTIPLIERS

        avg = sum(DAY_OF_WEEK_MULTIPLIERS.values()) / 7
        assert 0.9 <= avg <= 1.1

    def test_event_importance_levels_ordered(self) -> None:
        """Event importance levels should be ordered low to major."""
        from volume_forecast.config.constants import EVENT_IMPORTANCE

        assert EVENT_IMPORTANCE["low"] < EVENT_IMPORTANCE["medium"]
        assert EVENT_IMPORTANCE["medium"] < EVENT_IMPORTANCE["high"]
        assert EVENT_IMPORTANCE["high"] < EVENT_IMPORTANCE["major"]

    def test_customer_segments_sum_to_100(self) -> None:
        """Customer segment percentages should sum to 100."""
        from volume_forecast.config.constants import CUSTOMER_SEGMENTS

        total = sum(seg["percentage"] for seg in CUSTOMER_SEGMENTS.values())
        assert total == 100
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/config/constants.py`:

```python
"""Project constants for volume forecasting."""

from typing import Final

# Random seed for reproducibility
RANDOM_SEED: Final[int] = 42

# Date range for synthetic data generation (2 years)
DATA_START_DATE: Final[str] = "2023-01-01"
DATA_END_DATE: Final[str] = "2024-12-31"

# Forecast horizon
FORECAST_HORIZON_DAYS: Final[int] = 7

# Day-of-week volume multipliers (Monday=0, Sunday=6)
# Based on research: Saturday peak, Monday dip
DAY_OF_WEEK_MULTIPLIERS: Final[dict[int, float]] = {
    0: 0.85,   # Monday
    1: 0.80,   # Tuesday
    2: 0.85,   # Wednesday
    3: 0.90,   # Thursday
    4: 1.00,   # Friday
    5: 1.30,   # Saturday (peak)
    6: 1.10,   # Sunday
}

# Event importance levels and their volume multipliers
EVENT_IMPORTANCE: Final[dict[str, int]] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "major": 4,
}

EVENT_VOLUME_MULTIPLIERS: Final[dict[str, float]] = {
    "low": 1.2,
    "medium": 1.5,
    "high": 2.0,
    "major": 3.0,
}

# Customer segments with their characteristics
# Based on research: 70% casual, 20% regular, 7% heavy, 3% VIP
CUSTOMER_SEGMENTS: Final[dict[str, dict]] = {
    "casual": {
        "percentage": 70,
        "monthly_deposits": (1, 2),
        "avg_deposit_gbp": (10, 20),
        "monthly_logins": (5, 10),
    },
    "regular": {
        "percentage": 20,
        "monthly_deposits": (4, 8),
        "avg_deposit_gbp": (30, 50),
        "monthly_logins": (15, 30),
    },
    "heavy": {
        "percentage": 7,
        "monthly_deposits": (12, 20),
        "avg_deposit_gbp": (100, 200),
        "monthly_logins": (50, 80),
    },
    "vip": {
        "percentage": 3,
        "monthly_deposits": (20, 30),
        "avg_deposit_gbp": (500, 1000),
        "monthly_logins": (100, 150),
    },
}

# Base daily volumes (will be scaled by multipliers)
BASE_DAILY_LOGINS: Final[int] = 50000
BASE_DAILY_DEPOSITS: Final[int] = 8000
BASE_DAILY_DEPOSIT_VOLUME_GBP: Final[int] = 250000

# Seasonal multipliers (month 1-12)
MONTHLY_MULTIPLIERS: Final[dict[int, float]] = {
    1: 1.05,   # January - post-Christmas betting
    2: 0.95,   # February - quiet month
    3: 1.15,   # March - Cheltenham
    4: 1.10,   # April - Grand National
    5: 1.00,   # May - end of season
    6: 0.85,   # June - summer lull (unless tournament)
    7: 0.80,   # July - summer lull (unless tournament)
    8: 1.00,   # August - season start
    9: 1.05,   # September
    10: 1.05,  # October
    11: 1.05,  # November
    12: 1.15,  # December - Christmas period
}

# Column names
COL_DATE: Final[str] = "date"
COL_LOGINS: Final[str] = "daily_logins"
COL_DEPOSITS: Final[str] = "daily_deposits"
COL_DEPOSIT_VOLUME: Final[str] = "daily_deposit_volume_gbp"
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/config/constants.py tests/unit/test_config.py
git commit -m "feat: add configuration constants"
```

---

### Task 2.2: Create Settings with Pydantic

**Files:**
- Create: `src/volume_forecast/config/settings.py`
- Modify: `tests/unit/test_config.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_config.py`:

```python
class TestSettings:
    """Test suite for settings."""

    def test_settings_loads_defaults(self) -> None:
        """Settings should load with sensible defaults."""
        from volume_forecast.config.settings import Settings

        settings = Settings()
        assert settings.forecast_horizon == 7
        assert settings.random_seed == 42

    def test_settings_paths_are_pathlib(self) -> None:
        """Settings paths should be Path objects."""
        from pathlib import Path

        from volume_forecast.config.settings import Settings

        settings = Settings()
        assert isinstance(settings.data_dir, Path)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_config.py::TestSettings -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/config/settings.py`:

```python
"""Application settings using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent
    )
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))

    # Forecasting
    forecast_horizon: int = Field(default=7, ge=1, le=30)
    random_seed: int = Field(default=42)

    # Training
    min_training_days: int = Field(default=365)
    validation_folds: int = Field(default=5, ge=2)

    # External APIs
    football_api_key: str = Field(default="")
    cache_ttl_hours: int = Field(default=24)

    @property
    def raw_data_path(self) -> Path:
        """Path to raw data directory."""
        return self.project_root / self.data_dir / "raw"

    @property
    def processed_data_path(self) -> Path:
        """Path to processed data directory."""
        return self.project_root / self.data_dir / "processed"

    @property
    def external_data_path(self) -> Path:
        """Path to external data cache directory."""
        return self.project_root / self.data_dir / "external"


# Global settings instance
settings = Settings()
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_config.py::TestSettings -v
```

Expected: PASS

**Step 5: Update config __init__.py**

Update `src/volume_forecast/config/__init__.py`:

```python
"""Configuration module."""

from volume_forecast.config.constants import (
    BASE_DAILY_DEPOSITS,
    BASE_DAILY_LOGINS,
    COL_DATE,
    COL_DEPOSITS,
    COL_LOGINS,
    CUSTOMER_SEGMENTS,
    DAY_OF_WEEK_MULTIPLIERS,
    EVENT_IMPORTANCE,
    EVENT_VOLUME_MULTIPLIERS,
    FORECAST_HORIZON_DAYS,
    MONTHLY_MULTIPLIERS,
    RANDOM_SEED,
)
from volume_forecast.config.settings import Settings, settings

__all__ = [
    "Settings",
    "settings",
    "RANDOM_SEED",
    "FORECAST_HORIZON_DAYS",
    "DAY_OF_WEEK_MULTIPLIERS",
    "EVENT_IMPORTANCE",
    "EVENT_VOLUME_MULTIPLIERS",
    "CUSTOMER_SEGMENTS",
    "BASE_DAILY_LOGINS",
    "BASE_DAILY_DEPOSITS",
    "MONTHLY_MULTIPLIERS",
    "COL_DATE",
    "COL_LOGINS",
    "COL_DEPOSITS",
]
```

**Step 6: Commit**

```bash
git add src/volume_forecast/config/
git commit -m "feat: add pydantic settings"
```

---

## Phase 3: External Data Integration

### Task 3.1: Create Base API Client

**Files:**
- Create: `src/volume_forecast/external_data/base.py`
- Test: `tests/unit/test_external_data.py`

**Step 1: Write the failing test**

Create `tests/unit/test_external_data.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_external_data.py -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/external_data/base.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_external_data.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/external_data/base.py tests/unit/test_external_data.py
git commit -m "feat: add base API client with caching"
```

---

### Task 3.2: Create UK Holidays Client

**Files:**
- Create: `src/volume_forecast/external_data/holidays.py`
- Modify: `tests/unit/test_external_data.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_external_data.py`:

```python
from datetime import date


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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_external_data.py::TestUKHolidaysClient -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/external_data/holidays.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_external_data.py::TestUKHolidaysClient -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/external_data/holidays.py tests/unit/test_external_data.py
git commit -m "feat: add UK holidays API client"
```

---

### Task 3.3: Create Football Events Client

**Files:**
- Create: `src/volume_forecast/external_data/football.py`
- Modify: `tests/unit/test_external_data.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_external_data.py`:

```python
class TestFootballClient:
    """Test suite for football events client."""

    def test_determine_importance_champions_league_final(self) -> None:
        """Champions League final should be major importance."""
        from volume_forecast.external_data.football import FootballClient

        client = FootballClient()
        importance = client._determine_importance(
            competition="UEFA Champions League",
            stage="FINAL",
            home_team="Real Madrid",
            away_team="Liverpool",
        )
        assert importance == "major"

    def test_determine_importance_premier_league_regular(self) -> None:
        """Regular PL match should be medium importance."""
        from volume_forecast.external_data.football import FootballClient

        client = FootballClient()
        importance = client._determine_importance(
            competition="Premier League",
            stage="REGULAR_SEASON",
            home_team="Bournemouth",
            away_team="Brentford",
        )
        assert importance == "medium"

    def test_determine_importance_big_six_match(self) -> None:
        """Big six match should be high importance."""
        from volume_forecast.external_data.football import FootballClient

        client = FootballClient()
        importance = client._determine_importance(
            competition="Premier League",
            stage="REGULAR_SEASON",
            home_team="Manchester United",
            away_team="Liverpool",
        )
        assert importance == "high"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_external_data.py::TestFootballClient -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/external_data/football.py`:

```python
"""Football events API client using football-data.org."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

from volume_forecast.external_data.base import BaseAPIClient


class FootballClient(BaseAPIClient):
    """Client for football-data.org API."""

    API_URL = "https://api.football-data.org/v4"

    # Premier League "Big Six" teams
    BIG_SIX = {
        "Arsenal FC",
        "Chelsea FC",
        "Liverpool FC",
        "Manchester City FC",
        "Manchester United FC",
        "Tottenham Hotspur FC",
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
            if home_team in self.BIG_SIX and away_team in self.BIG_SIX:
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_external_data.py::TestFootballClient -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/external_data/football.py tests/unit/test_external_data.py
git commit -m "feat: add football events API client"
```

---

### Task 3.4: Create Static Event Calendars (Racing, Tennis, Boxing)

**Files:**
- Create: `src/volume_forecast/external_data/static_events.py`
- Modify: `tests/unit/test_external_data.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_external_data.py`:

```python
class TestStaticEventsCalendar:
    """Test suite for static events calendar."""

    def test_get_racing_events_2024(self) -> None:
        """Should return major racing events for 2024."""
        from volume_forecast.external_data.static_events import StaticEventsCalendar

        calendar = StaticEventsCalendar()
        events = calendar.get_racing_events(2024)

        # Should have Cheltenham, Grand National, Royal Ascot
        event_names = [e["name"] for e in events]
        assert any("Cheltenham" in name for name in event_names)
        assert any("Grand National" in name for name in event_names)
        assert any("Ascot" in name for name in event_names)

    def test_get_tennis_events(self) -> None:
        """Should return 4 Grand Slam events."""
        from volume_forecast.external_data.static_events import StaticEventsCalendar

        calendar = StaticEventsCalendar()
        events = calendar.get_tennis_events(2024)

        assert len(events) == 4  # 4 Grand Slams
        event_names = [e["name"] for e in events]
        assert any("Wimbledon" in name for name in event_names)

    def test_get_all_events_in_range(self) -> None:
        """Should return all events in a date range."""
        from volume_forecast.external_data.static_events import StaticEventsCalendar

        calendar = StaticEventsCalendar()
        events = calendar.get_events_in_range(
            date(2024, 3, 1),
            date(2024, 3, 31),
        )

        # March should have Cheltenham
        assert any("Cheltenham" in e["name"] for e in events)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_external_data.py::TestStaticEventsCalendar -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/external_data/static_events.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_external_data.py::TestStaticEventsCalendar -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/external_data/static_events.py tests/unit/test_external_data.py
git commit -m "feat: add static events calendar for racing, tennis, boxing"
```

---

### Task 3.5: Create Event Aggregator

**Files:**
- Create: `src/volume_forecast/external_data/aggregator.py`
- Modify: `tests/unit/test_external_data.py`
- Modify: `src/volume_forecast/external_data/__init__.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_external_data.py`:

```python
class TestEventAggregator:
    """Test suite for event aggregator."""

    def test_aggregates_all_event_types(self, tmp_path: Path) -> None:
        """Should aggregate events from all sources."""
        from volume_forecast.external_data.aggregator import EventAggregator

        aggregator = EventAggregator(cache_dir=tmp_path)
        events = aggregator.get_events(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 3, 31),
            include_football=False,  # Skip API call in test
        )

        # Should have racing (Cheltenham) and holiday events
        event_types = {e["event_type"] for e in events}
        assert "racing" in event_types or "holiday" in event_types

    def test_events_sorted_by_date(self, tmp_path: Path) -> None:
        """Events should be sorted by date."""
        from volume_forecast.external_data.aggregator import EventAggregator

        aggregator = EventAggregator(cache_dir=tmp_path)
        events = aggregator.get_events(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            include_football=False,
        )

        dates = [e["date"] for e in events]
        assert dates == sorted(dates)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_external_data.py::TestEventAggregator -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/external_data/aggregator.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_external_data.py::TestEventAggregator -v
```

Expected: PASS

**Step 5: Update module __init__.py**

Update `src/volume_forecast/external_data/__init__.py`:

```python
"""External data integration module."""

from volume_forecast.external_data.aggregator import EventAggregator
from volume_forecast.external_data.base import BaseAPIClient
from volume_forecast.external_data.football import FootballClient
from volume_forecast.external_data.holidays import UKHolidaysClient
from volume_forecast.external_data.static_events import StaticEventsCalendar

__all__ = [
    "BaseAPIClient",
    "UKHolidaysClient",
    "FootballClient",
    "StaticEventsCalendar",
    "EventAggregator",
]
```

**Step 6: Commit**

```bash
git add src/volume_forecast/external_data/
git commit -m "feat: add event aggregator"
```

---

## Phase 4: Synthetic Data Generation

### Task 4.1: Create Base Generator

**Files:**
- Create: `src/volume_forecast/data_generation/base.py`
- Test: `tests/unit/test_data_generation.py`

**Step 1: Write the failing test**

Create `tests/unit/test_data_generation.py`:

```python
"""Tests for data generation module."""

from datetime import date

import numpy as np
import pytest


class TestBaseGenerator:
    """Test suite for base generator."""

    def test_reproducible_with_seed(self) -> None:
        """Generator should produce reproducible results with same seed."""
        from volume_forecast.data_generation.base import BaseGenerator

        gen1 = BaseGenerator(seed=42)
        gen2 = BaseGenerator(seed=42)

        result1 = gen1.rng.random(10)
        result2 = gen2.rng.random(10)

        np.testing.assert_array_equal(result1, result2)

    def test_date_range_generation(self) -> None:
        """Should generate correct date range."""
        from volume_forecast.data_generation.base import BaseGenerator

        gen = BaseGenerator(seed=42)
        dates = gen.generate_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
        )

        assert len(dates) == 10
        assert dates[0] == date(2024, 1, 1)
        assert dates[-1] == date(2024, 1, 10)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_data_generation.py -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/data_generation/base.py`:

```python
"""Base generator class for synthetic data."""

from datetime import date, timedelta
from typing import Any

import numpy as np


class BaseGenerator:
    """Base class for data generators with reproducible random state."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_seed(self) -> None:
        """Reset the random number generator to initial seed."""
        self.rng = np.random.default_rng(self.seed)

    def generate_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """Generate a list of dates in a range.

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            List of dates.
        """
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def add_noise(
        self,
        values: np.ndarray,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """Add Gaussian noise to values.

        Args:
            values: Array of values.
            noise_level: Standard deviation as fraction of mean.

        Returns:
            Values with added noise.
        """
        noise = self.rng.normal(0, noise_level * np.mean(values), len(values))
        return values + noise

    def get_metadata(self) -> dict[str, Any]:
        """Get generator metadata for reproducibility.

        Returns:
            Dictionary of generator parameters.
        """
        return {
            "seed": self.seed,
            "generator_class": self.__class__.__name__,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_data_generation.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/data_generation/base.py tests/unit/test_data_generation.py
git commit -m "feat: add base data generator"
```

---

### Task 4.2: Create Volume Generator

**Files:**
- Create: `src/volume_forecast/data_generation/generator.py`
- Modify: `tests/unit/test_data_generation.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_data_generation.py`:

```python
import pandas as pd


class TestVolumeGenerator:
    """Test suite for volume generator."""

    def test_generates_correct_columns(self) -> None:
        """Should generate DataFrame with required columns."""
        from volume_forecast.data_generation.generator import VolumeGenerator

        gen = VolumeGenerator(seed=42)
        df = gen.generate(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert "date" in df.columns
        assert "daily_logins" in df.columns
        assert "daily_deposits" in df.columns
        assert "daily_deposit_volume_gbp" in df.columns

    def test_generates_correct_length(self) -> None:
        """Should generate correct number of rows."""
        from volume_forecast.data_generation.generator import VolumeGenerator

        gen = VolumeGenerator(seed=42)
        df = gen.generate(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert len(df) == 31

    def test_weekend_has_higher_volume(self) -> None:
        """Weekend should have higher average volume than weekdays."""
        from volume_forecast.data_generation.generator import VolumeGenerator

        gen = VolumeGenerator(seed=42)
        df = gen.generate(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        weekend_avg = df[df["day_of_week"] >= 5]["daily_logins"].mean()
        weekday_avg = df[df["day_of_week"] < 5]["daily_logins"].mean()

        assert weekend_avg > weekday_avg

    def test_values_are_positive(self) -> None:
        """All generated values should be positive."""
        from volume_forecast.data_generation.generator import VolumeGenerator

        gen = VolumeGenerator(seed=42)
        df = gen.generate(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert (df["daily_logins"] > 0).all()
        assert (df["daily_deposits"] > 0).all()
        assert (df["daily_deposit_volume_gbp"] > 0).all()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_data_generation.py::TestVolumeGenerator -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/data_generation/generator.py`:

```python
"""Main volume data generator."""

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volume_forecast.config.constants import (
    BASE_DAILY_DEPOSITS,
    BASE_DAILY_DEPOSIT_VOLUME_GBP,
    BASE_DAILY_LOGINS,
    COL_DATE,
    COL_DEPOSIT_VOLUME,
    COL_DEPOSITS,
    COL_LOGINS,
    DAY_OF_WEEK_MULTIPLIERS,
    EVENT_VOLUME_MULTIPLIERS,
    MONTHLY_MULTIPLIERS,
)
from volume_forecast.data_generation.base import BaseGenerator
from volume_forecast.external_data.aggregator import EventAggregator


class VolumeGenerator(BaseGenerator):
    """Generator for synthetic login and deposit volumes."""

    def __init__(
        self,
        seed: int = 42,
        base_logins: int = BASE_DAILY_LOGINS,
        base_deposits: int = BASE_DAILY_DEPOSITS,
        base_deposit_volume: int = BASE_DAILY_DEPOSIT_VOLUME_GBP,
        noise_level: float = 0.1,
    ) -> None:
        """Initialize the volume generator.

        Args:
            seed: Random seed.
            base_logins: Base daily login count.
            base_deposits: Base daily deposit count.
            base_deposit_volume: Base daily deposit volume in GBP.
            noise_level: Noise level as fraction of base values.
        """
        super().__init__(seed=seed)
        self.base_logins = base_logins
        self.base_deposits = base_deposits
        self.base_deposit_volume = base_deposit_volume
        self.noise_level = noise_level
        self._event_aggregator: EventAggregator | None = None

    @property
    def event_aggregator(self) -> EventAggregator:
        """Lazy-loaded event aggregator."""
        if self._event_aggregator is None:
            self._event_aggregator = EventAggregator()
        return self._event_aggregator

    def _get_day_multiplier(self, d: date) -> float:
        """Get day-of-week volume multiplier."""
        return DAY_OF_WEEK_MULTIPLIERS.get(d.weekday(), 1.0)

    def _get_month_multiplier(self, d: date) -> float:
        """Get monthly seasonality multiplier."""
        return MONTHLY_MULTIPLIERS.get(d.month, 1.0)

    def _get_event_multiplier(self, d: date, events_cache: dict) -> float:
        """Get event-based volume multiplier."""
        if d not in events_cache:
            return 1.0

        max_multiplier = 1.0
        for event in events_cache[d]:
            importance = event.get("importance", "low")
            multiplier = EVENT_VOLUME_MULTIPLIERS.get(importance, 1.0)
            max_multiplier = max(max_multiplier, multiplier)
        return max_multiplier

    def _build_events_cache(
        self, start_date: date, end_date: date
    ) -> dict[date, list[dict]]:
        """Pre-fetch and cache events for date range."""
        cache: dict[date, list[dict]] = {}
        try:
            events = self.event_aggregator.get_events(
                start_date, end_date, include_football=False
            )
            for event in events:
                event_date = event["date"]
                if event_date not in cache:
                    cache[event_date] = []
                cache[event_date].append(event)
        except Exception:
            pass  # Return empty cache on failure
        return cache

    def generate(
        self,
        start_date: date,
        end_date: date,
        include_events: bool = True,
    ) -> pd.DataFrame:
        """Generate synthetic volume data.

        Args:
            start_date: Start date for generation.
            end_date: End date for generation.
            include_events: Whether to include event effects.

        Returns:
            DataFrame with daily volumes.
        """
        dates = self.generate_date_range(start_date, end_date)
        n_days = len(dates)

        # Pre-cache events
        events_cache = {}
        if include_events:
            events_cache = self._build_events_cache(start_date, end_date)

        # Calculate multipliers for each day
        day_multipliers = np.array([self._get_day_multiplier(d) for d in dates])
        month_multipliers = np.array([self._get_month_multiplier(d) for d in dates])
        event_multipliers = np.array([
            self._get_event_multiplier(d, events_cache) for d in dates
        ])

        # Combined multiplier
        combined = day_multipliers * month_multipliers * event_multipliers

        # Generate base volumes with combined multipliers
        logins = self.base_logins * combined
        deposits = self.base_deposits * combined
        deposit_volume = self.base_deposit_volume * combined

        # Add correlated noise
        noise_logins = self.rng.normal(0, self.noise_level * self.base_logins, n_days)
        noise_deposits = self.rng.normal(0, self.noise_level * self.base_deposits, n_days)
        noise_volume = self.rng.normal(
            0, self.noise_level * self.base_deposit_volume, n_days
        )

        logins = np.maximum(logins + noise_logins, 100)  # Minimum floor
        deposits = np.maximum(deposits + noise_deposits, 10)
        deposit_volume = np.maximum(deposit_volume + noise_volume, 1000)

        # Apply login-deposit correlation (~70% correlation)
        deposit_adjustment = 0.7 * (logins / self.base_logins - 1)
        deposits = deposits * (1 + deposit_adjustment * 0.3)

        # Create DataFrame
        df = pd.DataFrame({
            COL_DATE: dates,
            COL_LOGINS: logins.astype(int),
            COL_DEPOSITS: deposits.astype(int),
            COL_DEPOSIT_VOLUME: deposit_volume.astype(int),
        })

        return df

    def save(
        self,
        df: pd.DataFrame,
        output_path: Path,
        include_metadata: bool = True,
    ) -> None:
        """Save generated data to CSV with optional metadata.

        Args:
            df: DataFrame to save.
            output_path: Output file path.
            include_metadata: Whether to save metadata JSON.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save CSV
        df.to_csv(output_path, index=False)

        # Save metadata
        if include_metadata:
            import json

            metadata = self.get_metadata()
            metadata.update({
                "start_date": df[COL_DATE].min().isoformat()
                if hasattr(df[COL_DATE].min(), "isoformat")
                else str(df[COL_DATE].min()),
                "end_date": df[COL_DATE].max().isoformat()
                if hasattr(df[COL_DATE].max(), "isoformat")
                else str(df[COL_DATE].max()),
                "n_rows": len(df),
                "base_logins": self.base_logins,
                "base_deposits": self.base_deposits,
                "noise_level": self.noise_level,
            })
            metadata_path = output_path.with_suffix(".meta.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

    def get_metadata(self) -> dict[str, Any]:
        """Get generator metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "base_logins": self.base_logins,
            "base_deposits": self.base_deposits,
            "base_deposit_volume": self.base_deposit_volume,
            "noise_level": self.noise_level,
        })
        return metadata
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_data_generation.py::TestVolumeGenerator -v
```

Expected: PASS

**Step 5: Update module __init__.py**

Update `src/volume_forecast/data_generation/__init__.py`:

```python
"""Synthetic data generation module."""

from volume_forecast.data_generation.base import BaseGenerator
from volume_forecast.data_generation.generator import VolumeGenerator

__all__ = [
    "BaseGenerator",
    "VolumeGenerator",
]
```

**Step 6: Commit**

```bash
git add src/volume_forecast/data_generation/
git commit -m "feat: add volume data generator"
```

---

## Phase 5: Feature Engineering

### Task 5.1: Create Base Transformer

**Files:**
- Create: `src/volume_forecast/features/base.py`
- Test: `tests/unit/test_features.py`

**Step 1: Write the failing test**

Create `tests/unit/test_features.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_features.py -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/features/base.py`:

```python
"""Base transformer class for feature engineering."""

from abc import ABC, abstractmethod
from typing import Any, Self

import pandas as pd


class BaseTransformer(ABC):
    """Abstract base class for feature transformers.

    Follows scikit-learn transformer interface for compatibility.
    """

    def __init__(self) -> None:
        """Initialize the transformer."""
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the transformer to data.

        Args:
            df: Input DataFrame.
            y: Target variable (optional, for compatibility).

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        pass

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Input DataFrame.
            y: Target variable (optional).

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df, y).transform(df)

    def get_feature_names(self) -> list[str]:
        """Get names of features created by this transformer.

        Returns:
            List of feature names.
        """
        return []

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters.

        Returns:
            Dictionary of parameters.
        """
        return {}
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_features.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/features/base.py tests/unit/test_features.py
git commit -m "feat: add base feature transformer"
```

---

### Task 5.2: Create Temporal Features

**Files:**
- Create: `src/volume_forecast/features/temporal.py`
- Modify: `tests/unit/test_features.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_features.py`:

```python
class TestTemporalFeatures:
    """Test suite for temporal features."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with dates."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "value": range(30),
        })

    def test_adds_day_of_week(self, sample_df: pd.DataFrame) -> None:
        """Should add day_of_week feature."""
        from volume_forecast.features.temporal import TemporalFeatures

        transformer = TemporalFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        assert "day_of_week" in result.columns
        assert result["day_of_week"].iloc[0] == 0  # Monday

    def test_adds_is_weekend(self, sample_df: pd.DataFrame) -> None:
        """Should add is_weekend feature."""
        from volume_forecast.features.temporal import TemporalFeatures

        transformer = TemporalFeatures(date_column="date")
        result = transformer.fit_transform(sample_df)

        assert "is_weekend" in result.columns
        # Jan 6, 2024 is Saturday
        saturday_idx = 5  # 6th day (0-indexed)
        assert result["is_weekend"].iloc[saturday_idx] == 1

    def test_adds_cyclical_features(self, sample_df: pd.DataFrame) -> None:
        """Should add sin/cos cyclical features."""
        from volume_forecast.features.temporal import TemporalFeatures

        transformer = TemporalFeatures(date_column="date", cyclical=True)
        result = transformer.fit_transform(sample_df)

        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_features.py::TestTemporalFeatures -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/features/temporal.py`:

```python
"""Temporal feature engineering."""

from typing import Any

import numpy as np
import pandas as pd

from volume_forecast.features.base import BaseTransformer


class TemporalFeatures(BaseTransformer):
    """Extract temporal features from date column."""

    def __init__(
        self,
        date_column: str = "date",
        cyclical: bool = True,
        include_payday: bool = True,
    ) -> None:
        """Initialize temporal features transformer.

        Args:
            date_column: Name of date column.
            cyclical: Whether to add cyclical sin/cos features.
            include_payday: Whether to add payday proximity feature.
        """
        super().__init__()
        self.date_column = date_column
        self.cyclical = cyclical
        self.include_payday = include_payday
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame by adding temporal features.

        Args:
            df: Input DataFrame with date column.

        Returns:
            DataFrame with added temporal features.
        """
        df = df.copy()
        dates = pd.to_datetime(df[self.date_column])

        # Basic temporal features
        df["day_of_week"] = dates.dt.dayofweek
        df["day_of_month"] = dates.dt.day
        df["month"] = dates.dt.month
        df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        df["is_month_start"] = dates.dt.is_month_start.astype(int)
        df["is_month_end"] = dates.dt.is_month_end.astype(int)

        self._feature_names = [
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "is_weekend",
            "is_month_start",
            "is_month_end",
        ]

        # Payday proximity (UK typically 25th or last working day)
        if self.include_payday:
            df["days_to_payday"] = df["day_of_month"].apply(
                lambda x: min(abs(25 - x), abs(25 - x + 30))
            )
            self._feature_names.append("days_to_payday")

        # Cyclical encoding
        if self.cyclical:
            # Day of week (period = 7)
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

            # Month (period = 12)
            df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
            df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

            # Week of year (period = 52)
            df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
            df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

            self._feature_names.extend([
                "day_of_week_sin",
                "day_of_week_cos",
                "month_sin",
                "month_cos",
                "week_sin",
                "week_cos",
            ])

        return df

    def get_feature_names(self) -> list[str]:
        """Get names of features created."""
        return self._feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters."""
        return {
            "date_column": self.date_column,
            "cyclical": self.cyclical,
            "include_payday": self.include_payday,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_features.py::TestTemporalFeatures -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/features/temporal.py tests/unit/test_features.py
git commit -m "feat: add temporal feature transformer"
```

---

### Task 5.3: Create Lag Features

**Files:**
- Create: `src/volume_forecast/features/lags.py`
- Modify: `tests/unit/test_features.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_features.py`:

```python
class TestLagFeatures:
    """Test suite for lag features."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "daily_logins": range(100, 130),
        })

    def test_creates_lag_columns(self, sample_df: pd.DataFrame) -> None:
        """Should create lag columns."""
        from volume_forecast.features.lags import LagFeatures

        transformer = LagFeatures(columns=["daily_logins"], lags=[1, 7])
        result = transformer.fit_transform(sample_df)

        assert "daily_logins_lag_1" in result.columns
        assert "daily_logins_lag_7" in result.columns

    def test_lag_values_correct(self, sample_df: pd.DataFrame) -> None:
        """Lag values should be shifted correctly."""
        from volume_forecast.features.lags import LagFeatures

        transformer = LagFeatures(columns=["daily_logins"], lags=[1])
        result = transformer.fit_transform(sample_df)

        # Row 1's lag_1 should equal row 0's value
        assert result["daily_logins_lag_1"].iloc[1] == sample_df["daily_logins"].iloc[0]

    def test_handles_missing_at_start(self, sample_df: pd.DataFrame) -> None:
        """Should handle missing values at series start."""
        from volume_forecast.features.lags import LagFeatures

        transformer = LagFeatures(columns=["daily_logins"], lags=[7])
        result = transformer.fit_transform(sample_df)

        # First 7 values should be NaN
        assert result["daily_logins_lag_7"].iloc[:7].isna().all()
        assert result["daily_logins_lag_7"].iloc[7:].notna().all()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_features.py::TestLagFeatures -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/features/lags.py`:

```python
"""Lag feature engineering."""

from typing import Any

import pandas as pd

from volume_forecast.features.base import BaseTransformer


class LagFeatures(BaseTransformer):
    """Create lag features from specified columns."""

    def __init__(
        self,
        columns: list[str],
        lags: list[int] | None = None,
    ) -> None:
        """Initialize lag features transformer.

        Args:
            columns: Columns to create lags for.
            lags: List of lag periods (default: [1, 7, 14, 21]).
        """
        super().__init__()
        self.columns = columns
        self.lags = lags or [1, 7, 14, 21]
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame by adding lag features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added lag features.
        """
        df = df.copy()
        self._feature_names = []

        for col in self.columns:
            if col not in df.columns:
                continue

            for lag in self.lags:
                feature_name = f"{col}_lag_{lag}"
                df[feature_name] = df[col].shift(lag)
                self._feature_names.append(feature_name)

        return df

    def get_feature_names(self) -> list[str]:
        """Get names of features created."""
        return self._feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters."""
        return {
            "columns": self.columns,
            "lags": self.lags,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_features.py::TestLagFeatures -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/features/lags.py tests/unit/test_features.py
git commit -m "feat: add lag feature transformer"
```

---

### Task 5.4: Create Rolling Features

**Files:**
- Create: `src/volume_forecast/features/rolling.py`
- Modify: `tests/unit/test_features.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_features.py`:

```python
class TestRollingFeatures:
    """Test suite for rolling features."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "daily_logins": [100] * 30,  # Constant for easy verification
        })

    def test_creates_rolling_mean(self, sample_df: pd.DataFrame) -> None:
        """Should create rolling mean columns."""
        from volume_forecast.features.rolling import RollingFeatures

        transformer = RollingFeatures(columns=["daily_logins"], windows=[7])
        result = transformer.fit_transform(sample_df)

        assert "daily_logins_rolling_mean_7" in result.columns

    def test_rolling_mean_value(self, sample_df: pd.DataFrame) -> None:
        """Rolling mean should be correct."""
        from volume_forecast.features.rolling import RollingFeatures

        transformer = RollingFeatures(columns=["daily_logins"], windows=[7])
        result = transformer.fit_transform(sample_df)

        # For constant values, rolling mean should equal the value
        assert result["daily_logins_rolling_mean_7"].iloc[7] == 100.0

    def test_creates_rolling_std(self, sample_df: pd.DataFrame) -> None:
        """Should create rolling std columns."""
        from volume_forecast.features.rolling import RollingFeatures

        transformer = RollingFeatures(
            columns=["daily_logins"], windows=[7], stats=["mean", "std"]
        )
        result = transformer.fit_transform(sample_df)

        assert "daily_logins_rolling_std_7" in result.columns
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_features.py::TestRollingFeatures -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/features/rolling.py`:

```python
"""Rolling window feature engineering."""

from typing import Any

import pandas as pd

from volume_forecast.features.base import BaseTransformer


class RollingFeatures(BaseTransformer):
    """Create rolling window statistics features."""

    def __init__(
        self,
        columns: list[str],
        windows: list[int] | None = None,
        stats: list[str] | None = None,
    ) -> None:
        """Initialize rolling features transformer.

        Args:
            columns: Columns to create rolling features for.
            windows: List of window sizes (default: [7, 14, 30]).
            stats: Statistics to compute (default: ["mean", "std"]).
        """
        super().__init__()
        self.columns = columns
        self.windows = windows or [7, 14, 30]
        self.stats = stats or ["mean", "std"]
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame by adding rolling features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added rolling features.
        """
        df = df.copy()
        self._feature_names = []

        for col in self.columns:
            if col not in df.columns:
                continue

            for window in self.windows:
                rolling = df[col].rolling(window=window, min_periods=1)

                for stat in self.stats:
                    feature_name = f"{col}_rolling_{stat}_{window}"

                    if stat == "mean":
                        df[feature_name] = rolling.mean()
                    elif stat == "std":
                        df[feature_name] = rolling.std()
                    elif stat == "min":
                        df[feature_name] = rolling.min()
                    elif stat == "max":
                        df[feature_name] = rolling.max()
                    elif stat == "sum":
                        df[feature_name] = rolling.sum()

                    self._feature_names.append(feature_name)

        return df

    def get_feature_names(self) -> list[str]:
        """Get names of features created."""
        return self._feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters."""
        return {
            "columns": self.columns,
            "windows": self.windows,
            "stats": self.stats,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_features.py::TestRollingFeatures -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/volume_forecast/features/rolling.py tests/unit/test_features.py
git commit -m "feat: add rolling feature transformer"
```

---

### Task 5.5: Create Feature Pipeline

**Files:**
- Create: `src/volume_forecast/features/pipeline.py`
- Modify: `tests/unit/test_features.py`
- Modify: `src/volume_forecast/features/__init__.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_features.py`:

```python
class TestFeaturePipeline:
    """Test suite for feature pipeline."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "daily_logins": range(1000, 1060),
            "daily_deposits": range(100, 160),
        })

    def test_pipeline_chains_transformers(self, sample_df: pd.DataFrame) -> None:
        """Pipeline should chain multiple transformers."""
        from volume_forecast.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            date_column="date",
            target_columns=["daily_logins"],
            lags=[1, 7],
            rolling_windows=[7],
        )
        result = pipeline.fit_transform(sample_df)

        # Should have temporal, lag, and rolling features
        assert "day_of_week" in result.columns
        assert "daily_logins_lag_1" in result.columns
        assert "daily_logins_rolling_mean_7" in result.columns

    def test_get_all_feature_names(self, sample_df: pd.DataFrame) -> None:
        """Should return all generated feature names."""
        from volume_forecast.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            date_column="date",
            target_columns=["daily_logins"],
        )
        result = pipeline.fit_transform(sample_df)
        feature_names = pipeline.get_feature_names()

        assert len(feature_names) > 0
        for name in feature_names:
            assert name in result.columns
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_features.py::TestFeaturePipeline -v
```

Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `src/volume_forecast/features/pipeline.py`:

```python
"""Feature engineering pipeline."""

from typing import Any, Self

import pandas as pd

from volume_forecast.features.base import BaseTransformer
from volume_forecast.features.lags import LagFeatures
from volume_forecast.features.rolling import RollingFeatures
from volume_forecast.features.temporal import TemporalFeatures


class FeaturePipeline(BaseTransformer):
    """Pipeline combining all feature transformers."""

    def __init__(
        self,
        date_column: str = "date",
        target_columns: list[str] | None = None,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        rolling_stats: list[str] | None = None,
        cyclical: bool = True,
    ) -> None:
        """Initialize the feature pipeline.

        Args:
            date_column: Name of date column.
            target_columns: Columns to create lag/rolling features for.
            lags: Lag periods to create.
            rolling_windows: Rolling window sizes.
            rolling_stats: Rolling statistics to compute.
            cyclical: Whether to add cyclical temporal features.
        """
        super().__init__()
        self.date_column = date_column
        self.target_columns = target_columns or ["daily_logins", "daily_deposits"]
        self.lags = lags or [1, 7, 14, 21]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.rolling_stats = rolling_stats or ["mean", "std"]
        self.cyclical = cyclical

        # Initialize transformers
        self._temporal = TemporalFeatures(
            date_column=date_column, cyclical=cyclical
        )
        self._lags = LagFeatures(columns=self.target_columns, lags=self.lags)
        self._rolling = RollingFeatures(
            columns=self.target_columns,
            windows=self.rolling_windows,
            stats=self.rolling_stats,
        )
        self._all_feature_names: list[str] = []

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit all transformers.

        Args:
            df: Input DataFrame.
            y: Target variable (optional).

        Returns:
            Self.
        """
        self._temporal.fit(df)
        self._lags.fit(df)
        self._rolling.fit(df)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all transformers.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame with all features.
        """
        df = df.copy()

        # Apply transformers in order
        df = self._temporal.transform(df)
        df = self._lags.transform(df)
        df = self._rolling.transform(df)

        # Collect feature names
        self._all_feature_names = (
            self._temporal.get_feature_names()
            + self._lags.get_feature_names()
            + self._rolling.get_feature_names()
        )

        return df

    def get_feature_names(self) -> list[str]:
        """Get all generated feature names.

        Returns:
            List of feature names.
        """
        return self._all_feature_names.copy()

    def get_params(self) -> dict[str, Any]:
        """Get pipeline parameters.

        Returns:
            Dictionary of parameters.
        """
        return {
            "date_column": self.date_column,
            "target_columns": self.target_columns,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "rolling_stats": self.rolling_stats,
            "cyclical": self.cyclical,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_features.py::TestFeaturePipeline -v
```

Expected: PASS

**Step 5: Update module __init__.py**

Update `src/volume_forecast/features/__init__.py`:

```python
"""Feature engineering module."""

from volume_forecast.features.base import BaseTransformer
from volume_forecast.features.lags import LagFeatures
from volume_forecast.features.pipeline import FeaturePipeline
from volume_forecast.features.rolling import RollingFeatures
from volume_forecast.features.temporal import TemporalFeatures

__all__ = [
    "BaseTransformer",
    "TemporalFeatures",
    "LagFeatures",
    "RollingFeatures",
    "FeaturePipeline",
]
```

**Step 6: Commit**

```bash
git add src/volume_forecast/features/
git commit -m "feat: add feature engineering pipeline"
```

---

## Phase 6: Models (Summary)

Due to length constraints, the following model tasks follow the same TDD pattern. Key files:

### Task 6.1: Base Model Interface
- `src/volume_forecast/models/base.py` - Abstract `BaseModel` class with `fit()`, `predict()`, `get_params()`

### Task 6.2: Baseline Models
- `src/volume_forecast/models/baselines.py` - `NaiveModel`, `SeasonalNaiveModel`, `MovingAverageModel`

### Task 6.3: Statistical Models
- `src/volume_forecast/models/statistical.py` - Wrapper for statsmodels SARIMA

### Task 6.4: Prophet Model
- `src/volume_forecast/models/prophet_model.py` - Prophet with holiday/event regressors

### Task 6.5: Tree Models
- `src/volume_forecast/models/tree_models.py` - LightGBM and XGBoost wrappers using skforecast

### Task 6.6: Ensemble Model
- `src/volume_forecast/models/ensemble.py` - Weighted ensemble combining multiple models

### Task 6.7: Model Registry
- `src/volume_forecast/models/registry.py` - Factory pattern for model instantiation

---

## Phase 7: Evaluation (Summary)

### Task 7.1: Metrics
- `src/volume_forecast/evaluation/metrics.py` - MAE, RMSE, MAPE, sMAPE functions

### Task 7.2: Walk-Forward Validation
- `src/volume_forecast/evaluation/validation.py` - `WalkForwardValidator` class

### Task 7.3: Model Comparison
- `src/volume_forecast/evaluation/comparison.py` - Benchmark runner and results aggregation

---

## Phase 8: Notebooks

Create Jupyter notebooks following the design document structure:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_external_data.ipynb`
3. `notebooks/03_feature_engineering.ipynb`
4. `notebooks/04_baseline_models.ipynb`
5. `notebooks/05_statistical_models.ipynb`
6. `notebooks/06_ml_models.ipynb`
7. `notebooks/07_model_comparison.ipynb`
8. `notebooks/08_final_forecast.ipynb`

---

## Verification Checklist

After completing all tasks:

```bash
# Run full test suite
uv run pytest tests/ -v --cov=src/volume_forecast

# Run linting
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Run type checking
uv run mypy src/

# Generate synthetic data
uv run python -c "
from volume_forecast.data_generation import VolumeGenerator
from datetime import date
gen = VolumeGenerator(seed=42)
df = gen.generate(date(2023, 1, 1), date(2024, 12, 31))
gen.save(df, 'data/raw/synthetic_volumes.csv')
print(f'Generated {len(df)} rows')
"

# Verify package imports
uv run python -c "
from volume_forecast.config import settings
from volume_forecast.data_generation import VolumeGenerator
from volume_forecast.external_data import EventAggregator
from volume_forecast.features import FeaturePipeline
print('All imports successful!')
"
```

---

**Plan complete and saved to `docs/plans/2026-01-04-volume-forecasting-implementation.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
