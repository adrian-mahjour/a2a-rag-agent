"""Tests the agentic rag graph"""

import asyncio
import os

import yaml
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph
from a2a_rag_agent.llm.llm_provider import LLMProvider
from a2a_rag_agent.utils.settings import LLMSettings
from a2a_rag_agent.graph.agentic_rag_graph import AgenticRagGraph, init_retriever_tool
from a2a_rag_agent.utils.langgraph_streaming import stream_graph_updates

load_dotenv()


async def init_graph() -> CompiledStateGraph:
    """initializes and compiles the agent graph"""
    llm_settings = LLMSettings()
    llm_provider = LLMProvider(settings=llm_settings)

    with open(os.environ["MODEL_CONFIG"], "r", encoding="utf-8") as f:  # TODO: env vars
        file_contents = yaml.safe_load(f)
        model_params = file_contents[llm_settings.LLM_MODEL_ID]
        embedding_model_params = file_contents[llm_settings.EMBEDDING_MODEL_ID]

    llm = await llm_provider.chat_model(llm_model_id=llm_settings.LLM_MODEL_ID, params=model_params)
    embedding_model = await llm_provider.embedding_model(
        embedding_model_id=llm_settings.EMBEDDING_MODEL_ID, params=embedding_model_params
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


async def main():
    graph = await init_graph()

    query = "What is the revenue that Software group generated?"
    print("invoking graph...")
    stream_graph_updates(graph=graph, user_input=query, config={})


if __name__ == "__main__":
    asyncio.run(main())
