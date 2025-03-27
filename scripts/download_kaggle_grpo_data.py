import kagglehub
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--data", type=str, required=False)
args = parser.parse_args()

# Download a single file.
if args.data:
    kagglehub.dataset_download(args.data)
if args.model:
    kagglehub.model_download(args.model)