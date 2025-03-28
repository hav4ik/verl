from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import pandas as pd
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--quant_path", type=str, required=True)
args = parser.parse_args()


# model_path = '/workspace/ft_7b/checkpoint-1144'
# quant_path = '/workspace/sft-7b-ep8'
model_path = args.model_path
quant_path = args.quant_path
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data="rbiswasfc/r1-7b", max_calib_seq_len=3072)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')