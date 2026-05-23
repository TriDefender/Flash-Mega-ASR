#!/bin/bash  
set -euo pipefail

# # wandb
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY="" # your wandb key
# export WANDB_PROJECT=""   
# export WANDB_ENTITY=""       
# export WANDB_MODE=online


# Data path will be replaced according to your actual requirements.
# TRAIN_JSONL=
# VAL_JSONL=
# OUT_DIR=
# LOG_FILE=
# RUN_NAME=

torchrun --nproc_per_node=2 A2S-SFT/finetune.py \
  --model_path Qwen3-ASR-1.7B \
  --train_file ${TRAIN_JSONL} \
  --eval_file ${VAL_JSONL} \
  --output_dir ${OUT_DIR} \
  --batch_size 8 \
  --grad_acc 8 \
  --lr 1e-6 \
  --lr_encoder 1e-6 \
  --lr_aligner 1e-6 \
  --lr_llm 1e-6 \
  --epochs 2 \
  --save_steps 200 \
  --save_total_limit 300 \
  --use_lora 1 \
  --lora_scope encoder_aligner/llm/all \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --run_name ${RUN_NAME} \
  --report_to wandb \
  2>&1 | tee -a ${LOG_FILE}