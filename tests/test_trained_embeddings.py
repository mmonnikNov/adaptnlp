import pytest
from adaptnlp import EasyWordEmbeddings, EasyStackedEmbeddings, EasyDocumentEmbeddings


@pytest.mark.skip(reason="No memory resources to support until circleci support is contacted")
def test_easy_word_embeddings():
    embeddings = EasyWordEmbeddings()
    embeddings.embed_text(text="Test", model_name_or_path="bert-base-cased")


@pytest.mark.skip(reason="No memory resources to support until circleci support is contacted")
def test_easy_stacked_embeddings():
    embeddings = EasyStackedEmbeddings("bert-base-cased", "xlnet-base-cased")
    embeddings.embed_text(text="Test")


@pytest.mark.skip(reason="No memory resources to support until circleci support is contacted")
def test_easy_document_embeddings():
    embeddings = EasyDocumentEmbeddings("bert-base-cased", "xlnet-base-cased")
    embeddings.embed_pool(text="Test")
