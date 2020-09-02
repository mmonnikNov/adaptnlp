from adaptnlp import EasySequenceClassifier


def test_easy_sequence_classifier():
    classifier = EasySequenceClassifier()
    classifier.tag_text(text="test", model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment")
