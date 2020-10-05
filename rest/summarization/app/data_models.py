from typing import List

from pydantic import BaseModel

# Summarization Request and Response
class SummarizationRequest(BaseModel):
    text: List[str]
    min_length: int = 100
    max_length: int = 500


class SummarizationResponse(BaseModel):
    text: List[str]
