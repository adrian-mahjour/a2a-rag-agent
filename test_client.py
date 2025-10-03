"""Makes a sample call to the A2A server"""

import asyncio
import logging

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory, create_text_message_object
from a2a.types import (
    AgentCard,
    TransportProtocol,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Get a logger instance
base_url = "http://localhost:10000"  # TODO: env var


async def fetch_agent_card(resolver: A2ACardResolver, base_url: str) -> AgentCard:
    """Fetches the agent's card from the server"""
    # Fetch Public Agent Card and Initialize Client
    final_agent_card_to_use: AgentCard | None = None

    try:
        logger.info(
            f"Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}"
        )
        _public_card = await resolver.get_agent_card()  # Fetches from default public path
        logger.info("Successfully fetched public agent card:")
        logger.info(_public_card.model_dump_json(indent=2, exclude_none=True))
        final_agent_card_to_use = _public_card
        logger.info("\nUsing PUBLIC agent card for client initialization (default).")

        if _public_card.supports_authenticated_extended_card:
            try:
                logger.info(
                    "\nPublic card supports authenticated extended card. "
                    "Attempting to fetch from: "
                    f"{base_url}{EXTENDED_AGENT_CARD_PATH}"
                )
                auth_headers_dict = {"Authorization": "Bearer dummy-token-for-extended-card"}
                _extended_card = await resolver.get_agent_card(
                    relative_card_path=EXTENDED_AGENT_CARD_PATH,
                    http_kwargs={"headers": auth_headers_dict},
                )
                logger.info("Successfully fetched authenticated extended agent card:")
                logger.info(_extended_card.model_dump_json(indent=2, exclude_none=True))
                final_agent_card_to_use = _extended_card  # Update to use the extended card
                logger.info(
                    "\nUsing AUTHENTICATED EXTENDED agent card for client " "initialization."
                )
            except Exception as e_extended:
                logger.warning(
                    f"Failed to fetch extended agent card: {e_extended}. "
                    "Will proceed with public card.",
                    exc_info=True,
                )
        elif _public_card:  # supports_authenticated_extended_card is False or None
            logger.info(
                "\nPublic card does not indicate support for an extended card. Using public card."
            )

        return final_agent_card_to_use

    except Exception as e:
        logger.error(f"Critical error fetching public agent card: {e}", exc_info=True)
        raise RuntimeError("Failed to fetch the public agent card. Cannot continue.") from e


async def main() -> None:

    async with httpx.AsyncClient() as httpx_client:

        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        # Fetch the agent card
        agent_card = await fetch_agent_card(resolver=resolver, base_url=base_url)

        # # Create A2A client with the agent card
        config = ClientConfig(
            httpx_client=httpx_client,
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
            ],
            use_client_preference=True,
        )

        factory = ClientFactory(config)
        client = factory.create(agent_card)

        logger.info("A2AClient initialized.")

        # Define the payload
        message_obj = create_text_message_object(content="What is the revenue of software group?")

        # Send the message
        stream_response = client.send_message(message_obj)

        # Iterate through the stream and print message
        async for chunk in stream_response:
            if chunk[-1] is not None:
                print(f"{chunk[-1].model_dump_json(indent=2, exclude_none=True)}")
            else:
                print(f"{chunk[0].model_dump_json(indent=2, exclude_none=True)}")
            print("\n")


if __name__ == "__main__":
    asyncio.run(main())
