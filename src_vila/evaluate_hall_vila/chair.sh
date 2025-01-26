#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate halva

export MASTER_ADDR=$(hostname)
echo "rank $SLURM_NODEID master: $MASTER_ADDR"
echo "rank $SLURM_NODEID Launching python script"

MASTER=`/bin/hostname -s`
MPORT=$(shuf -i 6000-9999 -n 1)

echo "Master: $MASTER"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "Port: $MPORT"
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

MODEL=$1
MODEL_BASE=$2
GPU_NUM=${3:-0}
MAX_NEW_TOKENS=1024
PROMPT_VERSION=v1


MODEL_DIR="/scratch/ssd004/scratch/pritam/OUTPUTS/HALVA/"

JOBID=${MODEL}
OUTDIR="/scratch/ssd004/scratch/pritam/OUTPUTS/HALVA/"${JOBID}
IMG_DIR="/scratch/ssd004/datasets/MSCOCO2014/val2014"
ANNO_DIR="/scratch/ssd004/scratch/pritam/datasets/coco/annotations/"

mkdir -p ${OUTDIR}/chair/
LOGFILE=${OUTDIR}/chair/eval.log

echo "logs are stored here: "${OUTDIR}

if [[ $MODEL_BASE == 'none' ]]; then
    # baseline
    MODEL_TYPE=$(echo "$MODEL" | sed 's/\//_/g')
    AFILE=${OUTDIR}/chair/answer_chair.jsonl
    RFILE=${OUTDIR}/chair/hall_words_chair.json

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall_vila.model_chair_loader \
        --model-path ${MODEL} \
        --image-folder ${IMG_DIR} \
        --answers-file ${AFILE} \
        --temperature 0 \
        --conv-mode ${PROMPT_VERSION} \
        --seed 42 \
        --num_samples 500 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --additional_input_prompt "Describe the image in detail." >> ${LOGFILE}

else
    # with lora
    MODEL_PATH=${MODEL_DIR}${MODEL}/
    AFILE=${OUTDIR}/chair/answer_chair.jsonl
    RFILE=${OUTDIR}/chair/hall_words_chair.json

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall_vila.model_chair_loader \
        --model-path ${MODEL_PATH} \
        --image-folder ${IMG_DIR} \
        --answers-file ${AFILE} \
        --temperature 0 \
        --conv-mode ${PROMPT_VERSION} \
        --model-base ${MODEL_BASE} \
        --seed 42 \
        --num_samples 500 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --additional_input_prompt "Describe the image in detail." >> ${LOGFILE}

fi


# some missing pkgs
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate mllm

# Calculate CHAIR using the generated jsonl file:
python eval_hall_vila/eval_chair.py \
    --cap_file ${AFILE} \
    --image_id_key question_id \
    --caption_key text \
    --coco_path ${ANNO_DIR} \
    --save_path ${RFILE} >> ${LOGFILE}

