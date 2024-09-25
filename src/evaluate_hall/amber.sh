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

MODE=$1 # 'g', 'd', 'de', 'da', 'dr'
MODEL=$2
MODEL_BASE=$3
GPU_NUM=${4:-0}
PROMPT_VERSION=v1
MAX_NEW_TOKENS=1024

JOBID=${MODEL} 
OUTDIR="/fs01/home/pritam/pritam_ssd004/OUTPUTS/HALVA/"${JOBID}
IMG_DIR="/fs01/home/pritam/pritam_ssd004/datasets/amber/image/"
ANNO_DIR="eval_hall/amber"

if [[ $MODE == 'a' ]]; then
    QFILE="eval_hall/amber/data/query/query_all.json"

elif [[ $MODE == 'g' ]]; then
    QFILE="eval_hall/amber/data/query/query_generative.json"

elif [[ $MODE == 'd' ]]; then
    QFILE="eval_hall/amber/data/query/query_discriminative.json"

elif [[ $MODE == 'de' ]]; then
    QFILE="eval_hall/amber/data/query/query_discriminative-existence.json"

elif [[ $MODE == 'da' ]]; then
    QFILE="eval_hall/amber/data/query/query_discriminative-attribute.json"

elif [[ $MODE == 'dr' ]]; then
    QFILE="eval_hall/amber/data/query/query_discriminative-relation.json"

else
    echo "unknow dataset "${MODE}
    exit
fi

mkdir -p ${OUTDIR}/amber/
LOGFILE=${OUTDIR}/amber/eval.log

echo "logs are stored here: "${OUTDIR}


if [[ $MODEL_BASE == 'none' ]]; then
    # baseline

    MODEL_TYPE=$(echo "$MODEL" | sed 's/\//_/g')
    AFILE=${OUTDIR}/amber/answer_amber_${MODE}.jsonl

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall.model_amber_loader \
    --model-path $MODEL \
    --question-file ${QFILE} \
    --image-folder ${IMG_DIR} \
    --answers-file ${AFILE} \
    --temperature 0 \
    --conv-mode ${PROMPT_VERSION} \
    --max_new_tokens ${MAX_NEW_TOKENS} >> ${LOGFILE}

else
    # with lora
    
    MODEL_DIR="/fs01/home/pritam/pritam_ssd004/OUTPUTS/HALVA/"
    MODEL_PATH=${MODEL_DIR}${MODEL}/

    AFILE=${OUTDIR}/amber/answer_amber_${MODE}.jsonl

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall.model_amber_loader \
    --model-path ${MODEL_PATH} \
    --question-file ${QFILE} \
    --image-folder ${IMG_DIR} \
    --answers-file ${AFILE} \
    --temperature 0 \
    --conv-mode ${PROMPT_VERSION} \
    --model-base ${MODEL_BASE} \
    --max_new_tokens ${MAX_NEW_TOKENS} >> ${LOGFILE}

fi

python eval_hall/amber/inference.py \
    --inference_data ${AFILE} \
    --evaluation_type ${MODE} \
    --anno_dir ${ANNO_DIR} >> ${LOGFILE}
