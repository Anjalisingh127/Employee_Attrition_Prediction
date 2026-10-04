"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe defaults for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    dataset_path: Path = Field(
        default=Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        alias="DATASET_PATH",
    )
    random_seed: int = Field(default=42, alias="RANDOM_SEED", ge=0)
    test_size: float = Field(default=0.20, alias="TEST_SIZE", gt=0.0, lt=1.0)


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object per process."""

    return Settings()
