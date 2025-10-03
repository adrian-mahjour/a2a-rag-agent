"""Tests the agentic rag graph"""

import asyncio

import yaml
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph

from a2a_rag_agent.genai.initialize import load_prompts
from a2a_rag_agent.graph.agentic_rag_graph import AgenticRagGraph, init_retriever_tool
from a2a_rag_agent.llm.llm_provider import LLMProvider
from a2a_rag_agent.utils.langgraph_streaming import stream_graph_updates
from a2a_rag_agent.utils.settings import AgentSettings, LLMProviderSettings


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


async def main():
    graph = await init_graph()

    query = "What is the revenue that Software group generated?"
    print("invoking graph...")
    stream_graph_updates(graph=graph, user_input=query, config={})


if __name__ == "__main__":
    asyncio.run(main())
