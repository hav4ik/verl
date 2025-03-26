python ../../scripts/peft_model_merger.py \
--local_dir /home/vu/data/verl/ckpts/DAPO/Deepseek-R1-1.5B-OpenRS-CoT4K-Lora-Cosine/global_step_50/actor/ \
--lora_r 64 \
--lora_alpha 128 \
--lora_target_modules "all-linear" \
--lora_bias "none"