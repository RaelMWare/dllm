#!/usr/bin/env bash
# Same recipe as fast_train.sh but starts from LLaDA-8B-INSTRUCT.
# Different output_dir so the Base FT checkpoint is not clobbered.
# Used for the controlled "does Instruct init help?" comparison.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /rds/general/user/rmw324/home/raels_playground/playground_1/dllm

accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml \
    --num_processes 1 \
    examples/llada/sft.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset_args /rds/general/user/rmw324/home/raels_playground/datasets/petfinder_llada_sft_v2 \
    --load_preprocessed_data True \
    --lora True \
    --r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,v_proj \
    --output_dir .models/llada-instruct-petfinder-fast \
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
