import pytest
from adaptnlp import EasySequenceClassifier

@pytest.mark.skip(
    reason="No memory resources to support until circleci support is contacted"
)
def test_easy_sequence_classifier():
    classifier = EasySequenceClassifier()
    classifier.tag_text(text="test", model_name_or_path="en-sentiment")
