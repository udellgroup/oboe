#!/usr/bin/env bash

# Shell script to generate error matrix, parallelizing across datasets.

DATA_DIR=$1
SAVE_DIR=$2
JSON_FILE=$3

if [[ -z ${2+x} ]]; then SAVE_DIR=""
fi

TIME=`date +%Y%m%d%H%M`

ls ${DATA_DIR}/*.csv | xargs -i --max-procs=90 bash -c \
"echo {}; python generate_vector.py "classification" {} --file=${JSON_FILE} --save_dir=${SAVE_DIR}/${TIME}"
