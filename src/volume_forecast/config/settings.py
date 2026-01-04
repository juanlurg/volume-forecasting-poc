"""Application settings using Pydantic."""

from pathlib import Path

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
