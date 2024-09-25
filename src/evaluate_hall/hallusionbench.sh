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

JOBID=${MODEL}
OUTDIR="/scratch/ssd004/scratch/pritam/OUTPUTS/HALVA/"${JOBID}
IMG_DIR='/fs01/home/pritam/pritam_ssd004/datasets/hallusion_bench'
TAG=$(date +"%m%d%H%M%S%4N")

# tag for 3 trials
VD=${OUTDIR}/hallusiobench/hallusion_output_vd_${TAG}.json
VS=${OUTDIR}/hallusiobench/hallusion_output_vs_${TAG}.json
AFILE=${OUTDIR}/hallusiobench/HallusionBenchOutput.json

mkdir -p ${OUTDIR}/hallusiobench/
LOGFILE=${OUTDIR}/hallusiobench/eval.log

GPT_MODEL='gpt-4-0613'
OPENAI_API_KEY="sk-mention-your-key"

if [[ $MODEL_BASE == 'none' ]]; then
    # baseline

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall.hallusion_bench.random_guess \
    --model-path ${MODEL} \
    --image-folder ${IMG_DIR} \
    --save_json_path_vd ${VD} \
    --save_json_path_vs ${VS} \
    --gpt_model ${GPT_MODEL} \
    --api_key ${OPENAI_API_KEY} \
    --output_file_name ${AFILE} >> ${LOGFILE}

else
    # with lora
    MODEL_DIR="/fs01/home/pritam/pritam_ssd004/OUTPUTS/HALVA/"
    MODEL_PATH=${MODEL_DIR}${MODEL}/

    CUDA_VISIBLE_DEVICES=$GPU_NUM python -m eval_hall.hallusion_bench.random_guess \
        --model-path ${MODEL_PATH} \
        --model-base ${MODEL_BASE} \
        --image-folder ${IMG_DIR} \
        --save_json_path_vd ${VD} \
        --save_json_path_vs ${VS} \
        --gpt_model ${GPT_MODEL} \
        --api_key ${OPENAI_API_KEY} \
        --output_file_name ${AFILE} >> ${LOGFILE}

fi
