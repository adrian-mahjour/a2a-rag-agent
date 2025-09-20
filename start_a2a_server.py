"""Starts the A2A server"""

import asyncio
import logging
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from a2a_rag_agent.rag_agent import RAGAgent
from a2a_rag_agent.agent_executor import LanggraphAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = "localhost"  # TODO: env var
PORT = 10000


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


async def get_agent_executor():
    return await LanggraphAgentExecutor.create()


def main(agent_executor):
    """Starts the A2A server."""
    try:
        # TODO: fix this with pydantic model
        # if os.getenv("model_source") == "google":
        #     if not os.getenv("GOOGLE_API_KEY"):
        #         raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")
        # else:
        #     if not os.getenv("TOOL_LLM_URL"):
        #         raise MissingAPIKeyError("TOOL_LLM_URL environment variable not set.")
        #     if not os.getenv("TOOL_LLM_NAME"):
        #         raise MissingAPIKeyError("TOOL_LLM_NAME environment not variable not set.")

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        skill = AgentSkill(
            id="report_qna",
            name="Q and A Earnings",
            description="Answers questions on financial reports",
            tags=["reports", "earnings"],
            examples=["What is the revenue that Software group generated?"],
        )
        agent_card = AgentCard(
            name="Earnings Agent",
            description="Helps with financial earnings reports",
            url=f"http://{HOST}:{PORT}/",
            version="1.0.0",
            default_input_modes=RAGAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=RAGAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client, config_store=push_config_store
        )
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

        uvicorn.run(server.build(), host=HOST, port=PORT)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    agent_executor = asyncio.run(get_agent_executor())
    main(agent_executor)
