#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate halva

export MASTER_ADDR=$(hostname)

MASTER=`/bin/hostname -s`
MPORT=$(shuf -i 6000-9999 -n 1)

echo "Master: $MASTER"
echo "Port: $MPORT"
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

wandb online

MODEL=$1
MODEL_BASE=$2
GPU_NUM=${3:-0}
PROMPT_VERSION=v1
MAX_NEW_TOKENS=1024

JOBID=${MODEL} 
DATA_ROOT="/fs01/home/pritam/pritam_ssd004/datasets/"
OUTDIR="/fs01/home/pritam/pritam_ssd004/OUTPUTS/HALVA/"${JOBID}

mkdir -p ${OUTDIR}/mmhal/
LOGFILE=${OUTDIR}/mmhal/eval.log
echo "logs are stored here: "${OUTDIR}

OPENAI_API_KEY='sk-mention-your-key'

if [[ $MODEL_BASE == 'none' ]]; then
    # baseline

    MODEL_TYPE=$(echo "$MODEL" | sed 's/\//_/g')
    AFILE=${OUTDIR}/mmhal/answer-file_mmhal.json
    GPT_EVAL_FILE1=${OUTDIR}/mmhal/review-file1_mmhal.json
    GPT_EVAL_FILE2=${OUTDIR}/mmhal/review-file2_mmhal.json
    GPT_EVAL_FILE3=${OUTDIR}/mmhal/review-file3_mmhal.json

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall.model_vqa_mmhal \
        --model-path ${MODEL} \
        --answers-file ${AFILE} \
        --conv-mode ${PROMPT_VERSION} \
        --temperature 0.0 \
        --test-prompt '' \
        --max_new_tokens $MAX_NEW_TOKENS \
        --data_root ${DATA_ROOT} >> ${LOGFILE}

else
    # with lora

    MODEL_DIR="/fs01/home/pritam/pritam_ssd004/OUTPUTS/HALVA/"
    MODEL_PATH=${MODEL_DIR}${MODEL}/

    AFILE=${OUTDIR}/mmhal/answer-file_mmhal.json
    GPT_EVAL_FILE1=${OUTDIR}/mmhal/review-file1_mmhal.json
    GPT_EVAL_FILE2=${OUTDIR}/mmhal/review-file2_mmhal.json
    GPT_EVAL_FILE3=${OUTDIR}/mmhal/review-file3_mmhal.json

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall.model_vqa_mmhal \
        --model-path ${MODEL_PATH} \
        --model-base ${MODEL_BASE} \
        --answers-file ${AFILE} \
        --conv-mode ${PROMPT_VERSION} \
        --temperature 0.0 \
        --test-prompt '' \
        --max_new_tokens $MAX_NEW_TOKENS \
        --data_root ${DATA_ROOT} >> ${LOGFILE}

fi

GPT='gpt-4-0125-preview'

# echo "first trial"
python eval_hall/eval_gpt_mmhal.py \
    --response ${AFILE} \
    --evaluation ${GPT_EVAL_FILE1} \
    --api-key ${OPENAI_API_KEY} \
    --gpt-model ${GPT}  >> ${LOGFILE}
    
python eval_hall/summarize_gpt_mmhal.py \
    --evaluation ${GPT_EVAL_FILE1} >> ${LOGFILE}

### for reporting run total 3 times

echo "second trial"
python eval_hall/eval_gpt_mmhal.py \
    --response ${AFILE} \
    --evaluation ${GPT_EVAL_FILE2} \
    --api-key ${OPENAI_API_KEY} \
    --gpt-model ${GPT}  >> ${LOGFILE}

python eval_hall/summarize_gpt_mmhal.py \
    --evaluation ${GPT_EVAL_FILE2} >> ${LOGFILE}

echo "third trial"
python eval_hall/eval_gpt_mmhal.py \
    --response ${AFILE} \
    --evaluation ${GPT_EVAL_FILE3} \
    --api-key ${OPENAI_API_KEY} \
    --gpt-model ${GPT}  >> ${LOGFILE}

python eval_hall/summarize_gpt_mmhal.py \
    --evaluation ${GPT_EVAL_FILE3} >> ${LOGFILE}
