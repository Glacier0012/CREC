#!/usr/bin/env bash

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
CONFIG=$1
GPUS=$2
EVAL_WEIGHTS=$3
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}

python3 -m torch.distributed.launch --nproc_per_node $GPUS --master_addr $ADDR --master_port $PORT \
tools/eval.py --config $CONFIG --eval-weights $EVAL_WEIGHTS