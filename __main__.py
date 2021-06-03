import os
import argparse

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