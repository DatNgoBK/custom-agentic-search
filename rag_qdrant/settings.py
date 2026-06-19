"""Centralized settings, loaded from environment / .env via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    url: str = Field(default="http://localhost:6333")
    api_key: str | None = Field(default=None)
    collection: str = Field(default="msb_report_2024")
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")


class EmbedSettings(BaseSettings):
    base_url: str = Field(default="http://localhost:8080")
    model: str = Field(default="intfloat/multilingual-e5-base")
    dim: int = Field(default=768, gt=0)
    api_key: str | None = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=".env", extra="ignore")


class RerankSettings(BaseSettings):
    base_url: str = Field(default="http://localhost:8081")
    model: str = Field(default="BAAI/bge-reranker-base")
    enabled: bool = Field(default=True)

    model_config = SettingsConfigDict(env_prefix="RERANK_", env_file=".env", extra="ignore")


class RetrievalSettings(BaseSettings):
    enable_sparse: bool = Field(default=True, alias="ENABLE_SPARSE")
    rrf_k: int = Field(default=60, alias="RRF_K", ge=1)
    dense_top_k: int = Field(default=20, alias="DENSE_TOP_K", ge=1)
    sparse_top_k: int = Field(default=20, alias="SPARSE_TOP_K", ge=1)
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K", ge=1)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class OpenVikingSettings(BaseSettings):
    data_path: str = Field(default="./data", alias="OV_DATA_PATH")
    project_name: str = Field(default="msb_demo", alias="OV_PROJECT_NAME")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class ObservabilitySettings(BaseSettings):
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    metrics_port: int = Field(default=9100, alias="METRICS_PORT", ge=1, le=65535)
    enable_otel: bool = Field(default=False, alias="ENABLE_OTEL")
    otel_endpoint: str = Field(
        default="http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class Settings:
    """Aggregate of all setting groups."""

    def __init__(self) -> None:
        self.qdrant = QdrantSettings()
        self.embed = EmbedSettings()
        self.rerank = RerankSettings()
        self.retrieval = RetrievalSettings()
        self.openviking = OpenVikingSettings()
        self.obs = ObservabilitySettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
