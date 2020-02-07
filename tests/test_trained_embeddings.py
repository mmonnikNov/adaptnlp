from adaptnlp import EasyWordEmbeddings, EasyStackedEmbeddings, EasyDocumentEmbeddings


def test_easy_word_embeddings():
    embeddings = EasyWordEmbeddings()

def test_easy_stacked_embeddings():
    embeddings = EasyStackedEmbeddings("bert", "roberta")


def test_easy_document_embeddings():
    embeddings = EasyDocumentEmbeddings("bert")
