import os
import argparse
import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from preprocess import tokenize
from test import test_load_docbin
#from visualize import draw
from visualize import draw_word_level
from clustering_algos import nearest_centroid_classifier
parser = argparse.ArgumentParser(description="Run BELIS")
parser.add_argument("--preprocess", help="Initiate preprocessing step", action="store_true")
parser.add_argument("--spacy_docbin_preprocess_path", help="Path to spacy docbin file from preprocess step. Required if --preprocess is not true.")
parser.add_argument("--spacy_vocab_preprocess_path", help="Path to spacy vocab file from preprocess step. Required if --preprocess is not true.")
parser.add_argument("--raw_preprocess_files", help="Path to dir with file(s) to be tokenized for the preprocess step. Required if --preprocess is true.")
parser.add_argument("--target_docbin_preprocess_path", help="where you would like the preprocess spacy docbin objects to go (include file name with .spacy at the end), default will save it to \"BELIS/datasets/docbin_preprocess.spacy\"")
parser.add_argument("--target_vocab_preprocess_path", help="where you would like the preprocess spacy vocab objects to go (include file name with .spacy at the end), default will save it to \"BELIS/datasets/vocab_preprocess.spacy\"")
parser.add_argument("--target_docbin_inference_path", help="where you would like the inference spacy docbin objects to go (include file name with .spacy at the end), default will save it to \"BELIS/datasets/docbin_inference.spacy\"")
parser.add_argument("--target_vocab_inference_path", help="where you would like the preprocess spacy vocab objects to go (include file name with .spacy at the end), default will save it to \"BELIS/datasets/vocab_preprocess.spacy\"")
parser.add_argument("--threads", help="How many threads to use for parallelization. Default is 1.", type=int)
parser.add_argument("--cross_validation", help="cross validate on data from docbin and vocab from preprocess",action="store_true")
parser.add_argument("--visualize", help="visualize clusters with umap",action="store_true")
parser.add_argument("--raw_inference_files", help="files to tokenize for inference.")
parser.add_argument("--inference", help="run inference on data from docbin and vocab",action="store_true")
parser.add_argument("--spacy_docbin_inference_path", help="Path to spacy docbin file of the data to run inference on. Required if --inference is true AND --raw_inference_files is not provided ")
parser.add_argument("--spacy_vocab_inference_path", help="Path to spacy vocab file of the data to run inference on. Required if --inference is true AND --raw_inference_files is not provided ")

args = parser.parse_args()
preprocess = args.preprocess
spacy_docbin_preprocess_path = args.spacy_docbin_preprocess_path
spacy_vocab_preprocess_path = args.spacy_vocab_preprocess_path
raw_preprocess_files = args.raw_preprocess_files
target_docbin_preprocess_path = args.target_docbin_preprocess_path
target_vocab_preprocess_path = args.target_vocab_preprocess_path
target_docbin_inference_path = args.target_docbin_inference_path
target_vocab_inference_path = args.target_vocab_inference_path
threads = args.threads
cross_validation = args.cross_validation
visualize = args.visualize
raw_inference_files = args.raw_inference_files
inference = args.inference
spacy_docbin_inference_path = args.spacy_docbin_inference_path
spacy_vocab_inference_path = args.spacy_vocab_inference_path
if target_docbin_preprocess_path == None:
    target_docbin_preprocess_path = "BELIS/datasets/docbin_preprocess.spacy"
if target_vocab_preprocess_path == None:
    target_vocab_preprocess_path="BELIS/datasets/vocab_preprocess.spacy"
if target_docbin_inference_path == None:
    target_docbin_inference_path="BELIS/datasets/docbin_inference.spacy"
if target_vocab_inference_path == None:
    target_vocab_inference_path = "BELIS/datasets/vocab_inference.spacy"
print(os.getcwd())

#Error handling.
if preprocess and raw_preprocess_files == None:
    print("Error: No argument specified for raw_preprocess_files, which is required for preprocess=True")
    exit()
elif (cross_validation or inference) and preprocess!=True and (spacy_vocab_preprocess_path == None or spacy_docbin_preprocess_path == None):
    print("Error: No argument specified for either spacy_vocab_preprocess_path or spacy_docbin_preprocess_path, which is required for cross_validation when preprocess=False")
    exit()
elif inference and raw_inference_files == None and (spacy_docbin_inference_path==None or spacy_vocab_inference_path == None):
    print("Error: No argument specified for raw_inference_files and either spacy_docbin_inference_path or spacy_vocab_inference_path, which is required to perform inference.")

if preprocess: #tokenize preprocessing code here
    if raw_preprocess_files.endswith("/") == False:
         raw_preprocess_files += "/"
    tokenize(raw_preprocess_files, target_docbin_preprocess_path, target_vocab_preprocess_path, False)
if cross_validation: #clustering the preprocess tokenizationresults
    print("Nearest Centroid Classification (cross_validation:)")
    docs = []
    if preprocess!=True:
        doc_bin = DocBin().from_disk(spacy_docbin_preprocess_path)
        vocab = Vocab().from_disk(spacy_vocab_preprocess_path)
        docs = list(doc_bin.get_docs(vocab))
    else:
        doc_bin = DocBin().from_disk(target_docbin_preprocess_path)
        vocab = Vocab().from_disk(target_vocab_preprocess_path)
        docs = list(doc_bin.get_docs(vocab))
    nearest_centroid_classifier(docs, True)

if inference:
    docs = []
    #first we need to tokenize the files we are running inference on, if we haven't already
    if raw_inference_files != None:
        tokenize(raw_inference_files, target_docbin_inference_path, target_vocab_inference_path, True)
    spacy_docbin_inference_path = target_docbin_inference_path
    spacy_vocab_inference_path = target_vocab_inference_path
    if preprocess:
        doc_bin = DocBin().from_disk(target_docbin_preprocess_path)
        vocab = Vocab().from_disk(target_vocab_preprocess_path)
        docs = list(doc_bin.get_docs(vocab))

    else:
        doc_bin = DocBin().from_disk(spacy_docbin_preprocess_path)
        vocab = Vocab().from_disk(spacy_vocab_preprocess_path)
        docs = list(doc_bin.get_docs(vocab))
    centroids, label_mapping =nearest_centroid_classifier(docs, False)
    #now that you have centroids and label_mapping, you can run an inference on the tokenized validation data!





if visualize:
    draw_word_level("C:/Users/nehav/Desktop/n2c2_100035_labeled_subwords_real.spacy")
