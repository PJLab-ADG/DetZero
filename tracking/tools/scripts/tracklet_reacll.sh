#!/usr/bin/env bash
set -x

SPLIT=$1
WORKERS=$2
DATA_PATH=$3
PY_ARGS=${@:4}

python -u eval_track.py \
       --split ${SPLIT} \
       --workers ${WORKERS} \
       --data_path ${DATA_PATH} \
       ${PY_ARGS}