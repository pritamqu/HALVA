#!/bin/bash

MODEL=$1 # pritamqu/halva13b-lora
MODEL_BASE=$2 # "liuhaotian/llava-v1.5-7b"

##### run amber 
bash src/vector/evaluate_hall/amber.sh g ${MODEL} ${MODEL_BASE} 0 &
bash src/vector/evaluate_hall/amber.sh da ${MODEL} ${MODEL_BASE} 1 &
bash src/vector/evaluate_hall/amber.sh dr ${MODEL} ${MODEL_BASE} 2 &
bash src/vector/evaluate_hall/amber.sh de ${MODEL} ${MODEL_BASE} 3 &
wait
# get amber f1 for all discriminateive tasks
bash src/vector/evaluate_hall/amber_f1.sh ${MODEL}

##### run chair
bash src/evaluate_hall/chair.sh ${MODEL} ${MODEL_BASE} 0

#### run mme
bash src/evaluate_hall/mme.sh ${MODEL} ${MODEL_BASE} 0

### run mmhal-bench
bash src/evaluate_hall/mmhal.sh ${MODEL} ${MODEL_BASE} 0

### run halusion-bench
bash src/evaluate_hall/hallusionbench.sh ${MODEL} ${MODEL_BASE} 0
