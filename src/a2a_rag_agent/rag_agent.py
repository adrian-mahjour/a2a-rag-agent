"""Defines the RAG Agent"""

from collections.abc import AsyncIterable
from typing import Any, Self

import yaml
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph

from a2a_rag_agent.graph.agentic_rag_graph import AgenticRagGraph, init_retriever_tool
from a2a_rag_agent.llm.llm_provider import LLMProvider
from a2a_rag_agent.utils.settings import LLMProviderSettings, AgentSettings
from a2a_rag_agent.genai.initialize import load_prompts


async def init_graph() -> CompiledStateGraph:
    """initializes and compiles the agent graph"""
    llm_provider_settings = LLMProviderSettings()
    agent_settings = AgentSettings()
    llm_provider = LLMProvider(settings=llm_provider_settings)

    # Initialize models and prompts
    with open(agent_settings.MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        file_contents = yaml.safe_load(f)
        model_params = file_contents[agent_settings.LLM_MODEL_ID]
        embedding_model_params = file_contents[agent_settings.EMBEDDING_MODEL_ID]

    llm = await llm_provider.chat_model(
        llm_model_id=agent_settings.LLM_MODEL_ID, params=model_params
    )
    embedding_model = await llm_provider.embedding_model(
        embedding_model_id=agent_settings.EMBEDDING_MODEL_ID, params=embedding_model_params
    )
    print("Initialized models")

    prompts = load_prompts(prompt_config_filepath=agent_settings.PROMPTS_CONFIG_PATH)

    retriever_tool = init_retriever_tool(
        input_file="data/report.txt",
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=150, chunk_overlap=24, length_function=len, is_separator_regex=False
        ),
        embedding_model=embedding_model,
        vec_store=InMemoryVectorStore,
    )
    rag_graph = AgenticRagGraph(llm=llm, retriever_tool=retriever_tool, prompts=prompts)
    print("Compiling graph...")
    graph = rag_graph.compile_graph()

    return graph


class RAGAgent:
    """Invokes or streams a given graph"""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph

    @classmethod
    async def create(cls) -> Self:
        """Creates an instance of the class"""
        graph = await init_graph()
        return cls(graph)

    def invoke(self, query: str, context_id: str) -> dict[str, Any]:
        """Defines syncronous invoke behaviour"""
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}
        response = self.graph.invoke({"messages": [("user", query)]}, config)
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": response.content,  # TODO: fix
        }

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        """Defines streaming behaviour"""
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": context_id}}

        async for item in self.graph.astream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]

            # If the AI has called a tool
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"AI is calling a tool: {message.tool_calls[0]}",
                }

            # If the message is a response from a tool
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"Tool response: {message.content}",
                }
        # Otherwise, the agent has returned its final response
        yield {
            "is_task_complete": False,
            "require_user_input": True,
            "content": message.content,
        }
