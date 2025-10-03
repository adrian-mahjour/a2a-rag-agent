"""AgentSettings"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Settings for the Agent"""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AGENT_", env_file_encoding="utf-8", extra="ignore"
    )

    LLM_MODEL_ID: str
    EMBEDDING_MODEL_ID: str
    MODEL_CONFIG_PATH: str
    PROMPTS_CONFIG_PATH: str
