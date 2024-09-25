#!/bin/bash

# ps
export HOME=/scratch/ssd004/scratch/anonymous/ 
source /h/anonymous/anaconda3/etc/profile.d/conda.sh
conda activate halva


EPOCH=1
LOSS_ALPHA=0.5
LR=5e-6
MODEL_NAME=liuhaotian/llava-v1.5-13b
JOBID="halva-13b-lora"

LORA_R=128
LORA_ALPHA=256
WD=0.
WARMUP_RATIO=0.03
MAX_LEN=2048
SAVE_STEPS=50000
SAVE_TOTAL_LIMIT=1
BATCH_SIZE=4
GRAD_ACC_STEPS=4

OUTDIR="/scratch/ssd004/scratch/anonymous/OUTPUTS/HALVA/"${JOBID}
IMG_DIR="default" # TODO: mention your root image path to access textvqa, gqa, vg, coco, ocr_vqa
REF_ANNO_FILE="data/llava_subset_ref.json"
ANNO_FILE="data/hvg.json"
mkdir -p ${OUTDIR}/train/

deepspeed train_halva.py \
    --lora_enable True \
    --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --mm_projector_lr 0 \
    --deepspeed src/json/zero3.json \
    --loss_alpha $LOSS_ALPHA \
    --model_name_or_path ${MODEL_NAME} \
    --version v1 \
    --data_path ${ANNO_FILE} \
    --ref_data_path ${REF_ANNO_FILE} \
    --image_folder ${IMG_DIR} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTDIR} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --weight_decay $WD \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to "wandb" \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --run_name ${JOBID} 

