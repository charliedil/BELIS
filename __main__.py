import os
import argparse
import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from preprocess import tokenize
parser = argparse.ArgumentParser(description="Run BELIS")
parser.add_argument("--preprocess", help="Initiate tokenization preprocessing", action="store_true")
parser.add_argument("--torch_dataset", help="Path to torch dataset file. Required if --preprocess is not true.")
parser.add_argument("--raw_files", help="Path to dir with file(s) for tokenizing. Required if --preprocess is true.")
parser.add_argument("--threads", help="How many threads to use for parallelization. Default is 1.", type=int)
args = parser.parse_args()
preprocess = args.preprocess
td_path = args.torch_dataset
raw_files = args.raw_files
threads = args.threads
print(os.getcwd())
#Error handling.
if preprocess and raw_files == None:
    print("Error: No argument specified for raw_files, which is required for preprocess=True")
    exit()
elif preprocess!=True and td_path == None:
    print("Error: No argument specified for torch_dataset, which is required for preprocess=False")
    exit()

if preprocess: #write actuall preprocessing code here
    if raw_files.endswith("/") == False:
         raw_files += "/"
    tokenize(raw_files)
else:
    print("Testing load capabilities: \nTODO: Modularize this code in separate cluster_algo file")
    doc_bin = DocBin().from_disk("BELIS/datasets/n2c2_100035.spacy")
    vocab = Vocab().from_disk("BELIS/datasets/n2c2_100035_vocab.spacy")
    docs = list(doc_bin.get_docs(vocab))
    for d in docs:
        for i in range(len(d.user_data["subwords"])):
            assert(len(d.user_data["subwords"][i])==len(d.user_data["subword_embeddings"][i]))
            #print(len(d.user_data["subwords"]))
            #print(len(d.user_data["subword_embeddings"]))
    #print(docs)
