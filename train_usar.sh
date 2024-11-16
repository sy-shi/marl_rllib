#!/bin/bash
SRC=$(pwd)
FILE="$SRC/config/usar_.yaml"

if [ ! -f "$FILE" ]; then
    echo "Error: $FILE does not exist."
    exit 1
fi

BASE_COMMAND="python main.py --config usar_ --mode tune --ckpt_freq 50 --timesteps_total 2000000"

for i in {1..5}; do
    NAME="usar_2ag_trial$i"
    $BASE_COMMAND --name=$NAME --model_path="policy_params/usar_2ag/$NAME"
done