from typing import List

from pydantic import BaseModel


# General Data Models
class Labels(BaseModel):
    value: str
    confidence: float


class Entities(BaseModel):
    text: str
    start_pos: int
    end_pos: int
    type: str
    confidence: float


# Sequence Classification
class SequenceClassificationRequest(BaseModel):
    text: str


class SequenceClassificationResponse(BaseModel):
    text: str
    labels: List[Labels] = []
    entities: List[Entities] = []
