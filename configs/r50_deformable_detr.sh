#!/usr/bin/env bash

set -x

EXP_DIR=AMself_output
PY_ARGS=${@:1}

python3 -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
