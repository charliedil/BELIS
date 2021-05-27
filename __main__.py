import os
import argparse

parser = argparse.ArgumentParser(description="Run BELIS")
parser.add_argument("--preprocess", help="Initiate tokenization preprocessing", action="store_true")
parser.add_argument("--torch_dataset", help="Path to torch dataset file. Required if --preprocess is not true.")
parser.add_argument("--raw_files", help="Path to dir with file(s) for tokenizing. Required if --preprocess is true.")
parser.add_argument("--threads", help="How many threads to use for parallelization. Default is 1.", type=int)
args = parser.parse_args()

#Error handling.
if args.preprocess and args.raw_files==None:
    print("Error: No argument specified for raw_files, which is required for preprocess=True")
    exit()
elif args.preprocess!=True and args.torch_dataset==None:
    print("Error: No argument specified for torch_dataset, which is required for preprocess=False")
    exit()

if args.preprocess: #write actuall preprocessing code here
    print("All good.")