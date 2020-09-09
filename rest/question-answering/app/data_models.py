from typing import List

from pydantic import BaseModel


# QA Label Object
class QASpanLabel(BaseModel):
    text: str
    probability: float
    start_logit: float
    end_logit: float
    start_index: int
    end_index: int

# Question Answering
class QuestionAnsweringRequest(BaseModel):
    query: str
    context: str
    top_n: int = 10


class QuestionAnsweringResponse(BaseModel):
    best_answer: str
    best_n_answers: List[QASpanLabel]
