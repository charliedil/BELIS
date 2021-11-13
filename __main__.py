import os
import argparse
import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from preprocess import tokenize_with_labels
from test import test_load_docbin
#from visualize import draw
from visualize import draw_word_level
from clustering_algos import nearest_centroid_classifier
parser = argparse.ArgumentParser(description="Run BELIS")
parser.add_argument("--preprocess", help="Initiate tokenization preprocessing", action="store_true")
parser.add_argument("--spacy_docbin_path", help="Path to spacy docbin file. Required if --preprocess is not true.")
parser.add_argument("--spacy_vocab_path", help="Path to spacy vocab file. Required if --preprocess is not true.")
parser.add_argument("--raw_init_files", help="Path to dir with file(s) for initial tokenizing. Required if --preprocess is true.")
parser.add_argument("--target_spacy_path", help="where should the spacy containers from preprocess go")
parser.add_argument("--threads", help="How many threads to use for parallelization. Default is 1.", type=int)
parser.add_argument("--cross_validation", help="cross validate on data from docbin and vocab",action="store_true")
parser.add_argument("--visualize", help="visualize clusters with umap",action="store_true")
parser.add_argument("--inference", help="run inference on data from docbin and vocab",action="store_true")




args = parser.parse_args()
preprocess = args.preprocess
vocab_path = args.spacy_vocab_path
docbin_path = args.spacy_docbin_path
raw_init_files = args.raw_init_files
threads = args.threads
cross_validation = args.cross_validation
visualize = args.visualize
inference = args.inference
target_spacy_path = args.target_spacy_path
if target_spacy_path == None:
    target_spacy_path="BELIS/datasets/"
print(os.getcwd())
#Error handling.
if preprocess and raw_init_files == None:
    print("Error: No argument specified for raw_files, which is required for preprocess=True")
    exit()
elif preprocess!=True and (vocab_path == None or docbin_path == None):
    print("Error: No argument specified for either --spacy_vocab_path or --spacy_docbin_path, which is required for preprocess=False")
    exit()

if preprocess: #write actuall preprocessing code here
    if raw_init_files.endswith("/") == False:
         raw_init_files += "/"
    tokenize_with_labels(raw_init_files, target_spacy_path)
if cross_validation: #clustering the words generated from preprocess, and visualizing them
    print("Nearest Centroid Classification:")
    docs = []
    if preprocess!=True:
        doc_bin = DocBin().from_disk(docbin_path)
        vocab = Vocab().from_disk(vocab_path)
        docs = list(doc_bin.get_docs(vocab))
    else:
        doc_bin = DocBin().from_disk(target_spacy_path+"n2c2_train_labeled.spacy")
        vocab = Vocab().from_disk(target_spacy_path+"n2c2_train_vocab.spacy")
        docs = list(doc_bin.get_docs(vocab))
    nearest_centroid_classifier(docs)
elif visualize:
    draw_word_level("C:/Users/nehav/Desktop/n2c2_100035_labeled_subwords_real.spacy")
    #print("Testing load capabilities: \nTODO: Modularize this code in separate cluster_algo file")

    #test_load_docbin("BELIS/datasets/n2c2_100035.spacy", "BELIS/datasets/n2c2_100035_vocab.spacy")
