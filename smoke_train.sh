#!/usr/bin/env bash
# Smoke training run: 5 optimiser steps on the small smoke dataset.
# Validates the training pipeline end-to-end before committing to the full run.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /rds/general/user/rmw324/home/raels_playground/playground_1/dllm

accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml \
    --num_processes 1 \
    examples/llada/sft.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Base \
    --dataset_args /rds/general/user/rmw324/home/raels_playground/datasets/petfinder_llada_sft_v2_smoke \
    --load_preprocessed_data True \
    --lora True \
    --output_dir .models/llada-base-petfinder-smoke \
    --max_steps 5 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 256 \
    --bf16 True \
    --logging_steps 1 \
    --save_strategy no
