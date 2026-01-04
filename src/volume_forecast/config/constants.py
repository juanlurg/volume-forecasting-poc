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
