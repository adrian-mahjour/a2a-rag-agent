import yaml
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from app.genai.models import PromptData

# TODO: create a setting pydantic for this


def load_prompts(prompt_config_filepath: str) -> dict[str, PromptData]:
    with open(prompt_config_filepath, "r", encoding="utf-8") as f:
        prompt_list = yaml.safe_load(f)

    initialized_prompts: dict[str, PromptData] = {
        prompt_dict["prompt_name"]: PromptData(**prompt_dict) for prompt_dict in prompt_list
    }

    return initialized_prompts


def create_chain(
    model: BaseLLM, prompt: PromptData, output_parser: BaseTransformOutputParser = StrOutputParser()
) -> Runnable:
    template = get_template(prompt=prompt)
    chain = template | model | output_parser

    return chain


def get_template(prompt: PromptData) -> PromptTemplate:
    return PromptTemplate(template=prompt.prompt_value)
