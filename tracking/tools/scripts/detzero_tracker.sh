#!/usr/bin/env bash
set -x

CFG_FILE=$1
DATA_PATH=$2
SET=$3
WORKERS=$4
PY_ARGS=${@:5}

python -u run_track.py \
	   --cfg_file ${CFG_FILE}  \
	   --data_path ${DATA_PATH} \
	   --split ${SET} \
	   --workers ${WORKERS} \
	   ${PY_ARGS}

