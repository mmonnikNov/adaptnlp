from typing import List

from pydantic import BaseModel

# Translation Request and Response
class TranslationRequest(BaseModel):
    text: List[str]


class TranslationResponse(BaseModel):
    text: List[str]
