#!/usr/bin/env bash
# Full training run: ~2,490 optimiser steps (3 epochs) on the real PetFinder SFT dataset.
# Run only after smoke_train.sh succeeds and you have rebuilt the dataset with SMOKE_TEST=False.
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
    --output_dir .models/llada-base-petfinder \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 256 \
    --bf16 True \
    --logging_steps 25 \
    --save_strategy epoch
