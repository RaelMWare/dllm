#!/usr/bin/env bash
# Fast training run: lighter LoRA config to fit in a single 8h reservation with margin.
# 1 epoch, rank 8 LoRA targeting only q_proj + v_proj. Expected ~1.5-2 hours.
#
# Quality vs full_train.sh: smaller adapter capacity, single pass through data.
# Good for a first-signal experiment ("does fine-tuning help?"). Scale up later
# (full_train.sh, more epochs) if results are promising.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /rds/general/user/rmw324/home/raels_playground/playground_1/dllm

accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml \
    --num_processes 1 \
    examples/llada/sft.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Base \
    --dataset_args /rds/general/user/rmw324/home/raels_playground/datasets/petfinder_llada_sft_v2 \
    --load_preprocessed_data True \
    --lora True \
    --r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,v_proj \
    --output_dir .models/llada-base-petfinder-fast \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 256 \
    --bf16 True \
    --logging_steps 25 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --eval_strategy no
