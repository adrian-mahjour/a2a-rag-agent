from typing import List, Optional

from pydantic import BaseModel


class PromptData(BaseModel):
    prompt_name: str
    description: Optional[str] = None
    prompt_value: str
    input_parameters: Optional[List[str]] = None
