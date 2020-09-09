import os
import logging
from typing import List

import adaptnlp

import uvicorn
from fastapi import FastAPI

from .data_models import (
    QuestionAnsweringRequest,
    QuestionAnsweringResponse,
)

app = FastAPI()

#####################
### Initialization###
#####################

# Initialize Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(process)d-%(levelname)s-%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Global Modules
_QA_MODEL = adaptnlp.EasyQuestionAnswering()

# Get Model Configurations From ENV VARS
_QUESTION_ANSWERING_MODEL = os.environ["QUESTION_ANSWERING_MODEL"]


# Event Handling
@app.on_event("startup")
async def initialize_nlp_task_modules():
    _QA_MODEL.predict_qa(
        query="-",
        context="______________________________________________________________________________",
        n_best_size=1,
        mini_batch_size=1,
        model_name_or_path=_QUESTION_ANSWERING_MODEL,
    )


######################
### AdaptNLP API ###
######################
@app.get("/")
async def root():
    return {"message": "Welcome to AdaptNLP"}


@app.post("/api/question-answering", response_model=QuestionAnsweringResponse)
async def question_answering(qa_request: QuestionAnsweringRequest):
    query = qa_request.query
    context = qa_request.context
    top_n = qa_request.top_n
    best_answer, best_n_answers = _QA_MODEL.predict_qa(
        query=query,
        context=context,
        n_best_size=top_n,
        mini_batch_size=1,
        model_name_or_path=_QUESTION_ANSWERING_MODEL,
    )
    payload = QuestionAnsweringResponse(
        best_answer=best_answer, best_n_answers=best_n_answers
    )
    return payload


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
