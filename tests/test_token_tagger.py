from adaptnlp import EasyTokenTagger


def test_easy_token_tagger():
    tagger = EasyTokenTagger()
    tagger.tag_text(text="test", model_name_or_path="ner-fast")
