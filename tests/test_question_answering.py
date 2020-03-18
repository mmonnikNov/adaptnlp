import pytest
from adaptnlp import EasyQuestionAnswering

@pytest.mark.skip(
    reason="No memory resources to support until circleci support is contacted"
)
def test_question_answering():
    qa_model = EasyQuestionAnswering()
    qa_model.predict_qa(query="Test", context="Test", n_best_size=1)
