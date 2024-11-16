#!/bin/bash
SRC=$(pwd)
FILE="$SRC/config/overcooked_.yaml"

if [ ! -f "$FILE" ]; then
    echo "Error: $FILE does not exist."
    exit 1
fi

BASE_COMMAND="python main.py --config overcooked_ --mode tune --ckpt_freq 40 --timesteps_total 2000000"

for i in {1..5}; do
    NAME="overcooked_3ag_trial$i"
    $BASE_COMMAND --name=$NAME --model_path="policy_params/overcooked_3ag/$NAME"
done