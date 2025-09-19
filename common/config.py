from pydantic import BaseSettings


class Settings(BaseSettings):
    azure_key_vault_name: str
    azure_foundry_endpoint: str
    azure_foundry_key: str

    class Config:
        env = ".env"

settings = Settings()