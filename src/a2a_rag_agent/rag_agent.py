"""Defines the RAG Agent"""

import os
from collections.abc import AsyncIterable
from typing import Any, Self

import yaml
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from a2a_rag_agent.graph.agentic_rag_graph import AgenticRagGraph, init_retriever_tool
from a2a_rag_agent.llm.llm_provider import LLMProvider
from a2a_rag_agent.llm.llm_settings import LLMSettings

memory = MemorySaver()  # TODO: what is this for?


async def init_graph() -> CompiledStateGraph:
    """initializes and compiles the agent graph"""
    llm_settings = LLMSettings()
    llm_provider = LLMProvider(settings=llm_settings)

    model_id = os.environ["LLM_MODEL_ID"]
    embedding_model_id = os.environ["EMBEDDING_MODEL_ID"]
    with open("config/models.yaml", "r", encoding="utf-8") as f:  # TODO: env vars
        model_params = yaml.safe_load(f)[model_id]
    with open("config/models.yaml", "r", encoding="utf-8") as f:
        embedding_model_params = yaml.safe_load(f)[embedding_model_id]

    llm = await llm_provider.chat_model(llm_model_id=model_id, params=model_params)
    embedding_model = await llm_provider.embedding_model(
        embedding_model_id=embedding_model_id, params=embedding_model_params
    )
    print("Initialized models")

    retriever_tool = init_retriever_tool(
        input_file="data/report.txt",
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=150, chunk_overlap=24, length_function=len, is_separator_regex=False
        ),
        embedding_model=embedding_model,
        vec_store=InMemoryVectorStore,
    )
    rag_graph = AgenticRagGraph(llm=llm, retriever_tool=retriever_tool)
    print("Compiling graph...")
    graph = rag_graph.compile_graph()

    return graph


class RAGAgent:
    """RAG Agent"""

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
