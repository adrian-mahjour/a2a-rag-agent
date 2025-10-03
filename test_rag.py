"""Tests the agentic rag graph"""

import asyncio
import os

import yaml
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from a2a_rag_agent.llm.llm_provider import LLMProvider
from a2a_rag_agent.llm.llm_settings import LLMSettings
from a2a_rag_agent.graph.agentic_rag_graph import AgenticRagGraph, init_retriever_tool
from a2a_rag_agent.utils.langgraph_streaming import stream_graph_updates

load_dotenv()


async def main():
    llm_settings = LLMSettings()
    llm_provider = LLMProvider(settings=llm_settings)

    model_id = os.environ["LLM_MODEL_ID"]
    embedding_model_id = os.environ["EMBEDDING_MODEL_ID"]
    with open("config/models.yaml", "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f)[model_id]
    with open("config/models.yaml", "r", encoding="utf-8") as f:
        embedding_model_params = yaml.safe_load(f)[embedding_model_id]

    llm = await llm_provider.chat_model(llm_model_id=model_id, params=model_params)
    embedding_model = await llm_provider.embedding_model(
        embedding_model_id=embedding_model_id, params=embedding_model_params
    )
    print("initialized models")

    retriever_tool = init_retriever_tool(
        input_file="data/report.txt",
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=150, chunk_overlap=24, length_function=len, is_separator_regex=False
        ),
        embedding_model=embedding_model,
        vec_store=InMemoryVectorStore,
    )
    rag_graph = AgenticRagGraph(llm=llm, retriever_tool=retriever_tool)
    print("compiling graph...")
    graph = rag_graph.compile_graph()

    query = "What is the revenue that Software group generated?"
    print("invoking graph...")
    stream_graph_updates(graph=graph, user_input=query, config={})


if __name__ == "__main__":
    asyncio.run(main())
