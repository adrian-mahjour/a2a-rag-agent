"""Agent executor responsible for translating LangGraph output to A2A protocol"""

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from a2a_rag_agent.rag_agent import RAGAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanggraphAgentExecutor(AgentExecutor):
    """LanggraphAgentExecutor."""

    def __init__(self, agent):
        self.agent = agent

    @classmethod
    async def create(cls):
        # Perform asynchronous operations here
        agent = await RAGAgent.create()
        return cls(agent)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            async for item in self.agent.stream(query, task.context_id):
                # Update the TaskState and artifacts based on the dict returned by the agent

                # example items dict {
                #     "is_task_complete": False,
                #     "require_user_input": False,
                #     "content": "Looking up the exchange rates...",
                # }
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]

                if not is_task_complete and not require_user_input:
                    # Update the task status
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            text=item["content"],
                            context_id=task.context_id,
                            task_id=task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            text=item["content"],
                            context_id=task.context_id,
                            task_id=task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item["content"]))],
                        name="agent_result",
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
