"""OllamaSettings"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Ollama related settings"""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LLM_OLLAMA_", env_file_encoding="utf-8", extra="ignore"
    )

    BASE_URL: str
