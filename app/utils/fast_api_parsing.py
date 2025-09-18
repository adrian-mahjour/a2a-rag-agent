import json
from typing import Optional
from uuid import uuid4

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.graph.state import CompiledStateGraph


def serialize_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    raise TypeError(
        f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
    )


async def generate_chat_responses(
    graph: CompiledStateGraph, message: str, checkpoint_id: Optional[str] = None
):
    is_new_conversation = checkpoint_id is None

    if is_new_conversation:
        new_checkpoint_id = str(uuid4())

        config = {"configurable": {"thread_id": new_checkpoint_id}}

        # Initialize with first message
        events = graph.astream_events({"messages": [HumanMessage(content=message)]}, config=config)

        # Fiest send the checkpoint ID
        yield f'data: {{"type":"checkpoint", "checkpoint_id": "{new_checkpoint_id}}}\n\n'

    else:
        config = {"configurable": {"thread_id": checkpoint_id}}

        # Continue existing conversation
        events = graph.astream_events({"messages": [HumanMessage(content=message)]}, config=config)

    for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_content = serialize_ai_message_chunk(event["data"]["chunk"])

            # Escape single quotes and newlines for JSON parsing
            yield f'data: {{"type":"content", "content": "{json.dumps(chunk_content)}}}\n\n'

        elif event_type == "on_chat_model_end":
            # Check if there are tool calls
            tool_calls = (
                event["data"]["output"].tool_calls
                if hasattr(event["data"]["output"], "tool_calls")
                else []
            )

            # If the too was called, yield the tool name
            if tool_calls:
                yield f"data: {{\"type\": \"tool_call\", \"name\": \"{tool_calls[0]['name']}\"}}\n\n"

        elif event_type == "on_tool_end":
            # If the tool returned as response, yield the tool response
            output = event["data"]["output"]

            yield f'data: {{"type":"tool_result", "result": "{json.dumps(output.content)}}}\n\n'

    # When done, send end event
    yield 'data: {"type": "end"}\n\n'
