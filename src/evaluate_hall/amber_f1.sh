#!/bin/bash

DIRNAME=$1

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate halva


ROOT="/fs01/home/pritam/pritam_ssd004/OUTPUTS/HALVA"
ANNO_DIR="eval_hall/amber"

python eval_hall/amber/merge.py ${DIRNAME}

AFILE=${ROOT}/${DIRNAME}/amber/answer_amber_d.jsonl
python eval_hall/amber/inference.py \
    --inference_data ${AFILE} \
    --evaluation_type d \
    --anno_dir ${ANNO_DIR} >> ${ROOT}/${DIRNAME}/amber/eval.log
