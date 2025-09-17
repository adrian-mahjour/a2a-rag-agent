import asyncio

from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM

from a2a_rag_agent.llm.llm_backend import LLMBackend
from a2a_rag_agent.llm.llm_settings import LLMSettings
from a2a_rag_agent.llm.ollama import OllamaSettings
from a2a_rag_agent.utils.singleton_meta import SingletonMeta


class LLMProvider(metaclass=SingletonMeta):
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings()
        self.cache = {}
        self.lock = asyncio.Lock()

    def _get_cache_key(self, model_type: str, model_id: str, params: dict = None) -> str:
        sorted_tuples = [("model_type", model_type), ("model_id", model_id)]
        if params:
            sorted_tuples += sorted(params.items())
        return str(sorted_tuples)

    @staticmethod
    async def test_connection(
        credentials: OllamaSettings, llm_model_id: str, params: dict
    ) -> BaseLLM | None:
        if isinstance(credentials, OllamaSettings):
            return OllamaLLM(model=llm_model_id, base_url=credentials.BASE_URL, **params)
        return None

    async def llm_model(self, llm_model_id: str, params: dict) -> BaseLLM:
        cache_key = self._get_cache_key(
            model_type="llm_model", model_id=llm_model_id, params=params
        )

        if cache_key in self.cache:
            return self.cache[cache_key]

        async with self.lock:
            if cache_key in self.cache:
                return self.cache[cache_key]

            model = await self._llm_model(llm_model_id=llm_model_id, params=params)
            self.cache[cache_key] = model
            return model

    async def chat_model(self, llm_model_id: str, params: dict) -> BaseChatModel:
        cache_key = self._get_cache_key(
            model_type="chat_llm_model", model_id=llm_model_id, params=params
        )

        if cache_key in self.cache:
            return self.cache[cache_key]

        async with self.lock:
            if cache_key in self.cache:
                return self.cache[cache_key]

            model = await self._chat_llm_model(llm_model_id=llm_model_id, params=params)
            self.cache[cache_key] = model
            return model

    async def embedding_model(self, embedding_model_id: str, params: dict) -> Embeddings:
        cache_key = self._get_cache_key(
            model_type="embedding_model", model_id=embedding_model_id, params=params
        )

        if cache_key in self.cache:
            return self.cache[cache_key]

        async with self.lock:
            if cache_key in self.cache:
                return self.cache[cache_key]

            model = await self._embedding_model(
                embedding_model_id=embedding_model_id, params=params
            )
            self.cache[cache_key] = model
            return model

    # Define the various LLM, Chat, and Embeddings models for each provider below

    async def _llm_model(self, llm_model_id: str, params: dict) -> BaseLLM:
        if self.settings.BACKEND == LLMBackend.OLLAMA:
            return OllamaLLM(
                model=llm_model_id, base_url=self.settings.backend_setting.BASE_URL, **params
            )
        raise ValueError("Unsupported value for llm backend")

    async def _chat_llm_model(self, llm_model_id: str, params: dict) -> BaseLLM:
        if self.settings.BACKEND == LLMBackend.OLLAMA:
            return ChatOllama(
                model=llm_model_id, base_url=self.settings.backend_setting.BASE_URL, **params
            )

        raise ValueError("Unsupported value for llm backend")

    async def _embedding_model(self, embedding_model_id: str, params: dict) -> BaseLLM:
        if self.settings.BACKEND == LLMBackend.OLLAMA:
            return OllamaEmbeddings(
                model=embedding_model_id, base_url=self.settings.backend_setting.BASE_URL, **params
            )

        raise ValueError("Unsupported value for llm backend")
