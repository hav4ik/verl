import kagglehub
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# Download a single file.
kagglehub.dataset_download(args.data)
kagglehub.model_download(args.model)