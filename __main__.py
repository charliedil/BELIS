import os
import argparse
import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from preprocess import tokenize
from test import test_load_docbin
from visualize import draw
from clustering_algos import nearest_centroid_classifier
parser = argparse.ArgumentParser(description="Run BELIS")
parser.add_argument("--preprocess", help="Initiate tokenization preprocessing", action="store_true")
parser.add_argument("--spacy_docbin_path", help="Path to spacy docbin file. Required if --preprocess is not true.")
parser.add_argument("--spacy_vocab_path", help="Path to spacy vocab file. Required if --preprocess is not true.")
parser.add_argument("--raw_files", help="Path to dir with file(s) for tokenizing. Required if --preprocess is true.")
parser.add_argument("--threads", help="How many threads to use for parallelization. Default is 1.", type=int)

args = parser.parse_args()
preprocess = args.preprocess
vocab_path = args.spacy_vocab_path
docbin_path = args.spacy_docbin_path
raw_files = args.raw_files
threads = args.threads
print(os.getcwd())
#Error handling.
if preprocess and raw_files == None:
    print("Error: No argument specified for raw_files, which is required for preprocess=True")
    exit()
elif preprocess!=True and (vocab_path == None or docbin_path == None):
    print("Error: No argument specified for either --spacy_vocab_path or --spacy_docbin_path, which is required for preprocess=False")
    exit()

if preprocess: #write actuall preprocessing code here
    if raw_files.endswith("/") == False:
         raw_files += "/"
    tokenize(raw_files)
else:
    print("Nearest Centroid Classification:")
    doc_bin = DocBin().from_disk(docbin_path)
    vocab = Vocab().from_disk(vocab_path)
    docs = list(doc_bin.get_docs(vocab))
    nearest_centroid_classifier(docs)
    #draw("C:/Users/nehav/Desktop/n2c2_100035_labeled_subwords_real.spacy")
    #print("Testing load capabilities: \nTODO: Modularize this code in separate cluster_algo file")

    #test_load_docbin("BELIS/datasets/n2c2_100035.spacy", "BELIS/datasets/n2c2_100035_vocab.spacy")
