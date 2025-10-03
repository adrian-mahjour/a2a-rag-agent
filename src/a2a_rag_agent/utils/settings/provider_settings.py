"""LLMSettings"""

from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict

from a2a_rag_agent.llm.llm_backend import LLMBackend
from a2a_rag_agent.utils.settings import OllamaSettings


class LLMProviderSettings(BaseSettings):
    """Defines the settings for the LLM Service"""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LLM_", env_file_encoding="utf-8", extra="ignore"
    )

    BACKEND: LLMBackend

    @cached_property
    def backend_setting(self) -> OllamaSettings | None:
        """returns the model settings used to create langchain clients"""
        if self.BACKEND == LLMBackend.OLLAMA:
            return OllamaSettings()
        return None
