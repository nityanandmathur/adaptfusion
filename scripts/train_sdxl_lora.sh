#!bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="/home/btech/nityanand.mathur/cityscapes/lora"
export OUTPUT_DIR="../models/cityscapes-sdxl-lora-r4-i1000"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

knockknock slack --webhook-url https://hooks.slack.com/services/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\
  --channel domain-adaptation\
  accelerate launch --multi_gpu ../src/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="sks scene" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="a car in sks scene" \
  --validation_epochs=100 \
  --seed="0" \
  --push_to_hub
