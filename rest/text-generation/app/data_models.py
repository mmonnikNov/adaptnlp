from typing import List

from pydantic import BaseModel

# Summarization Request and Response
class TextGenerationRequest(BaseModel):
    text: str
    num_tokens_to_produce: int = 50


class TextGenerationResponse(BaseModel):
    text: List[str]
