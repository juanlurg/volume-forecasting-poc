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
