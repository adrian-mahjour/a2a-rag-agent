"""AgenticRAGGraph"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages.base import BaseMessage
from langchain_core.tools.simple import Tool
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters.base import TextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from a2a_rag_agent.genai.models import PromptData

load_dotenv()


def init_retriever_tool(
    input_file: str,
    text_splitter: TextSplitter,
    embedding_model: Embeddings,
    vec_store: VectorStore,
) -> Tool:
    # TODO: this should be in it's own module?
    """Creates a retriever tool"""
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Create documents
    documents = text_splitter.create_documents([content])

    # Create vector store
    vector_store = vec_store.from_documents(documents=documents, embedding=embedding_model)

    # Create retreiver tool
    # TODO: make customizable
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="retreive_report_info",
        description="Search and retrun information about reports",
    )

    return retriever_tool


class GradeDocuments(BaseModel):
    """Create documents using a binary score for relevance check"""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class AgenticRagGraph:
    """Defines the graph structure of the agnet"""

    def __init__(self, llm: BaseLLM, retriever_tool: Tool, prompts: dict[str, PromptData]) -> None:
        self.agent_model = llm
        self.retriever_tool = retriever_tool
        self.prompts = prompts

    def generate_query_or_respond(self, state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Call the model the generate a response based on the current state.
        Given the question, it will decide to retrieve using retriever tool, or simply repond
        to the user"""

        response = self.agent_model.bind_tools([self.retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    def grade_documents(
        self, state: MessagesState
    ) -> Literal["generate_answer"] | Literal["rewrite_question"]:
        """Determine wheter the retrived documents are relevant to the question"""

        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = self.prompts["grade_prompt"].prompt_value.format(
            question=question, context=context
        )

        response = self.agent_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )

        score = response.binary_score

        if score == "yes":
            print("=== [DECISION: DOCS RELEVANT] ===")
            return "generate_answer"

        print("=== [DECISION: DOCS NOT RELEVANT] ===")
        return "rewrite_question"

    def rewrite_question(self, state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Rewrites the original user question"""
        question = state["messages"][0].content

        prompt = self.prompts["rewrite_prompt"].prompt_value.format(question=question)

        response = self.agent_model.invoke([{"role": "user", "content": prompt}])

        return {"messages": [response]}

    def generate_answer(self, state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Generates an answer"""
        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = self.prompts["generate_prompt"].prompt_value.format(
            question=question, context=context
        )

        response = self.agent_model.invoke([{"role": "user", "content": prompt}])

        return {"messages": [response]}

    def compile_graph(self) -> CompiledStateGraph:
        """Compiles the AgenticRAG graph"""
        graph_builder = StateGraph(MessagesState)

        # Define the nodes
        graph_builder.add_node(self.generate_query_or_respond)
        graph_builder.add_node("retrieve", ToolNode([self.retriever_tool]))
        graph_builder.add_node(self.rewrite_question)
        graph_builder.add_node(self.generate_answer)

        # Define edges
        graph_builder.add_edge(START, "generate_query_or_respond")
        graph_builder.add_conditional_edges(
            "generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END}
        )
        graph_builder.add_conditional_edges("retrieve", self.grade_documents)
        graph_builder.add_edge("generate_answer", END)
        graph_builder.add_edge("rewrite_question", "generate_query_or_respond")

        # TODO: compile with memory
        # checkpointer = InMemorySaver()
        # graph = graph_builder.compile(checkpointer=checkpointer)

        graph = graph_builder.compile()

        return graph
