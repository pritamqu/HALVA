#!/bin/bash

# MODEL=$1 # pritamqu/halva13b-lora
# MODEL_BASE=$2 # "liuhaotian/llava-v1.5-7b"

# MODEL="Efficient-Large-Model/VILA1.5-3b"
# MODEL_BASE="none"


MODEL="Halva-VILA-1.5-3b-lora-alpha-0.4-lr-1e-5"
MODEL_BASE="Efficient-Large-Model/VILA1.5-3b"

##### run amber 
bash src/evaluate_hall_vila/amber.sh g ${MODEL} ${MODEL_BASE} 0

bash src/evaluate_hall_vila/amber.sh da ${MODEL} ${MODEL_BASE} 1 &
bash src/evaluate_hall_vila/amber.sh dr ${MODEL} ${MODEL_BASE} 2 &
bash src/evaluate_hall_vila/amber.sh de ${MODEL} ${MODEL_BASE} 3 &
wait
# get amber f1 for all discriminateive tasks
bash src/evaluate_hall_vila/amber_f1.sh ${MODEL}

##### run chair
bash src/evaluate_hall_vila/chair.sh ${MODEL} ${MODEL_BASE} 0

#### run mme
bash src/evaluate_hall_vila/mme.sh ${MODEL} ${MODEL_BASE} 0

# ### run mmhal-bench
# bash src/evaluate_hall_vila/mmhal.sh ${MODEL} ${MODEL_BASE} 0

# ### run halusion-bench
# bash src/evaluate_hall_vila/hallusionbench.sh ${MODEL} ${MODEL_BASE} 0
