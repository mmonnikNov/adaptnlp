from adaptnlp import EasyTranslator


def test_easy_translator():
    translator = EasyTranslator()
    translator.translate(text="Testing summarizer")
