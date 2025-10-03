"""Basic Streamlit UI for testing the A2A Agent"""

import json
import os
import logging
from uuid import uuid4

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Get a logger instance


# TODO: why is request returning ping?
def get_data_from_streaming_api(api_url: str, prompt: str):
    payload = {
        "id": str(uuid4()),
        "jsonrpc": "2.0",
        "method": "message/stream",
        "params": {
            "message": {
                "kind": "message",
                "messageId": str(uuid4()),
                "parts": [{"kind": "text", "text": prompt}],
                "role": "user",
            }
        },
    }
    with requests.post(api_url, json=payload, stream=True) as r:
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                # TODO: best way to parse and display the response? tools, tool responses, task updates, etc.
                # if "data: " in chunk:
                #     data = chunk.split("data: ")[1]  # remove the "data: " string
                #     parsed_data = json.loads(data)
                #     pretty_json_string = json.dumps(parsed_data, indent=2)
                #     yield pretty_json_string
                # else:
                yield chunk


# Initialize Streamlit page
st.set_page_config(page_title="A2A RAG Agent Chat", layout="centered")
st.title("A2A RAG Agent Interaction")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What was the revenue of the software group?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the message
    with st.chat_message("ai"):
        st.write_stream(
            get_data_from_streaming_api(
                f"http://{os.environ["A2A_SERVER_HOST"]}:{os.environ["A2A_SERVER_PORT"]}", prompt
            )
        )
