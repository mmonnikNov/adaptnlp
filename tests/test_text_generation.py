
from adaptnlp import EasyTextGenerator


def test_easy_generator():
    generator = EasyTextGenerator()
    generator.generate(text="Testing generator")
