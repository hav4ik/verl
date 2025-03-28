#!/usr/bin/env bash
set -euxo pipefail
export VLLM_USE_V1=1

project_name='DAPO-14B'
exp_name='sft14b-v6-dapo-lora-exp02'
adv_estimator=grpo
kl_coef=0.0
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
overlong_buffer_len=$((1024 * 1))
overlong_buffer_enable=True
use_token_level_loss=True
enable_filter_groups=True

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
RUNTIME_ENV=${RUNTIME_ENV:-"./verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/workspace"}
KAGGLEHUB_CACHE=${KAGGLEHUB_CACHE:-"/workspace/kagglehub"}
MODEL_PATH=${MODEL_PATH:-"${KAGGLEHUB_CACHE}/models/conjuring92/aimo-sft-14b/Transformers/default/6"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${KAGGLEHUB_CACHE}/datasets/chankhavu/aimo-grpo-train-data-v00/versions/1/train-grpo-exp00-x10.parquet"}
TEST_FILE=${TEST_FILE:-"${KAGGLEHUB_CACHE}/datasets/chankhavu/aimo-grpo-train-data-v00/versions/1/valid-40-probs-x8.parquet"}

# Algorithm
## Train
learning_rate=2e-6
max_prompt_length=$((512 * 1))
max_response_length=$((1024 * 8))
max_packed_length=$((1024 * 64))  # For sequence packing
gen_prompt_bsz=48  # Should be equal to train_prompt_bsz if enable_filter_groups is False
train_prompt_bsz=32  # Real batch size that will be picked for training (x n_resp_per_prompt)
train_prompt_mini_bsz=16  # ppo mini batch size (real bs is this x n_resp_per_prompt)
n_resp_per_prompt=10  # Real train prompt batch size = train_prompt_bsz * n_resp_per_prompt
ppo_repeat_batch=2  # Perform 2 "epochs" of training on the same batch
rewards_manager=naive  # wither naive (pure DAPO) or dapo_openrs (DAPO with format and Cosine length loss)
## Validation
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=4  # >1 to enable Ulysses
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
n_gpus_per_node=4
gen_tp=2

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${PWD}" \
#     -- python3 -m verl.trainer.main_ppo \
VLLM_USE_V1=1 python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.fill_to_train_bsz=True \
    algorithm.filter_groups.drop_last_mini_batch=True \
    +algorithm.adv_scale_reward=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_packed_length} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=256 \
    +actor_rollout_ref.model.use_dora=False \
    +actor_rollout_ref.model.use_rslora=True \
    actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_repeat_batch} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_token_level_loss=${use_token_level_loss} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen2DecoderLayer,Qwen2Attention,Qwen2MLP] \
    custom_reward_function.overlong_buffer.enable=${overlong_buffer_enable} \
    custom_reward_function.overlong_buffer.len=${overlong_buffer_len} \
    custom_reward_function.overlong_buffer.penalty_factor=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes="${NNODES}" \
    +trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.critic_warmup=0 \
    reward_model.reward_manager=${rewards_manager}
