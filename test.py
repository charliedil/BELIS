import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab

def test_load_docbin(docbin_path, vocab_path):
    doc_bin = DocBin().from_disk("BELIS/datasets/n2c2_100035.spacy")
    vocab = Vocab().from_disk("BELIS/datasets/n2c2_100035_vocab.spacy")
    docs = list(doc_bin.get_docs(vocab))
    for d in docs:
        for i in range(len(d.user_data["subwords"])):
            assert(len(d.user_data["subwords"][i])==len(d.user_data["subword_embeddings"][i]))
