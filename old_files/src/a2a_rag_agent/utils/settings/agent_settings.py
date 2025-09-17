from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AGENT_", env_file_encoding="utf-8", extra="ignore"
    )

    CONFIG_PATH: str
    GREETINGS_PATH: str
    MCP_URL: Optional[str] = None
    REPORT_STYLE_PATH: Optional[str] = None
