from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict

from a2a_rag_agent.llm.llm_backend import LLMBackend
from a2a_rag_agent.llm.ollama import OllamaSettings


class LLMSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LLM_", env_file_encoding="utf-8", extra="ignore"
    )

    BACKEND: LLMBackend = LLMBackend.OLLAMA  # hardcoded to ollama

    @cached_property
    def backend_setting(self) -> OllamaSettings | None:
        if self.BACKEND == LLMBackend.OLLAMA:
            return OllamaSettings()
        return None
