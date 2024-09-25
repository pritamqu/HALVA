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

MODEL=$1 # 
MODEL_BASE=$2 #
GPU_NUM=${3:-0}
PROMPT_VERSION=v1
MAX_NEW_TOKENS=1024

JOBID=${MODEL}
MODEL_TYPE=$(echo "$MODEL" | sed 's/\//_/g')

OUTDIR="/scratch/ssd004/scratch/pritam/OUTPUTS/HALVA/"${JOBID}
MME_ROOT="/fs01/home/pritam/pritam_ssd004/datasets/MME"
QFILE=${MME_ROOT}/llava_mme.jsonl
AFILE=${OUTDIR}/mme/answer_mme.jsonl
IMG_DIR="/scratch/ssd004/scratch/pritam/datasets/MME/MME_Benchmark_release_version"
mkdir -p ${OUTDIR}/mme/
LOGFILE=${OUTDIR}/mme/eval.log


if [[ $MODEL_BASE == 'none' ]]; then
    # baseline
    
    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall.model_vqa_loader \
    --model-path ${MODEL} \
    --question-file ${QFILE} \
    --image-folder ${IMG_DIR} \
    --answers-file ${AFILE} \
    --temperature 0 \
    --conv-mode ${PROMPT_VERSION} \
    --max_new_tokens ${MAX_NEW_TOKENS} >> ${LOGFILE}

else
    # with lora
    MODEL_DIR="/scratch/ssd004/scratch/pritam/OUTPUTS/HALVA/"
    MODEL_PATH=${MODEL_DIR}${MODEL}/

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall.model_vqa_loader \
        --model-path ${MODEL_PATH} \
        --model-base ${MODEL_BASE} \
        --question-file ${QFILE} \
        --image-folder ${IMG_DIR} \
        --answers-file ${AFILE} \
        --temperature 0 \
        --conv-mode ${PROMPT_VERSION} \
        --max_new_tokens ${MAX_NEW_TOKENS} >> ${LOGFILE}

fi

python src/evaluate_hall/convert_answer_to_mme.py \
    --data_path ${IMG_DIR} \
    --result_file ${AFILE} >> ${LOGFILE}

cd /fs01/home/pritam/pritam_ssd004/datasets/MME/eval_tool/ ## TODO: replace this with your path of MME/eval_tool/
python calculation.py --results_dir ${OUTDIR}/mme/answers >> ${LOGFILE}
# DONOT ADD / after answers
