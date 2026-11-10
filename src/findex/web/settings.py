from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    # Конфіг лише з оточення: INDEX_PATH, LOG_LEVEL, SNIPPET_WIDTH, TEXT_CACHE.
    # Регістр імен не важливий; невідомі змінні оточення ігноруються.
    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    index_path: Path = Path("index.bin")
    log_level: LogLevel = "INFO"
    snippet_width: int = Field(default=80, ge=10, le=400)
    text_cache: int = Field(
        default=64, ge=0, description="Скільки текстів документів тримати в пам'яті"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
