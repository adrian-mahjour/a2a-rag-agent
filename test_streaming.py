import asyncio
import os
import yaml
from dotenv import load_dotenv

from a2a_rag_agent.llm.llm_provider import LLMProvider
from a2a_rag_agent.llm.llm_settings import LLMSettings
from a2a_rag_agent.genai.initialize import load_prompts, create_chain
from langchain_core.language_models.llms import BaseLLM

load_dotenv()


async def main():
    llm_settings = LLMSettings()
    llm_provider = LLMProvider(settings=llm_settings)

    prompts = load_prompts(prompt_config_filepath="config/prompts.yaml")

    model_id = os.environ["LLM_MODEL_ID"]
    with open("config/models.yaml", "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f)[model_id]

    llm: BaseLLM = await llm_provider.llm_model(llm_model_id=model_id, params=model_params)

    chain = create_chain(model=llm, prompt=prompts["test_prompt"])
    print("initialized")

    print("calling model...")
    # response = chain.invoke({})  # empty dict, bc no inputs to prompt
    # print(response)

    async for chunk in llm.astream("Write me a 1 verse song about sparkling water."):
        print(chunk, end="|", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
