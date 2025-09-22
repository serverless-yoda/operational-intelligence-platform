# common/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Load from .env; ignore stray keys; allow case-insensitive env var lookup
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,   # your .env is lowercase; this makes env lookups flexible
    )

    # App metadata
    app_name: str = "Operational Engagement Platform"

    # These are REQUIRED and map directly to your .env keys (lowercase matches fields)
    azure_key_vault_name: str
    azure_foundry_endpoint: str            # secret NAME in Key Vault
    azure_foundry_key: str                 # secret NAME in Key Vault
    azure_foundry_inference_url: str       # secret NAME in Key Vault

    # Optional/defaulted fields (add if/when you need them)
    azure_foundry_deployment: str | None = None
    azure_foundry_deployment_embed: str | None = None
    # azure_foundry_api_version: str = "2024-05-01-preview"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Construct with no args so pydantic-settings reads .env / env automatically
    return Settings()

settings = get_settings()
