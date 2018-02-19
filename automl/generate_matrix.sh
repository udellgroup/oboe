#!/usr/bin/env bash

# Shell script to generate error matrix, parallelizing across datasets.

DATA_DIR=$1
SAVE_DIR=$2
JSON_FILE=$3

if [[ -z ${2+x} ]]; then SAVE_DIR=""
fi

TIME=`date +%Y%m%d%H%M`

for file in ${DATA_DIR}/*.csv;
do
    python generate_vector.py "classification" ${file} --file=${JSON_FILE} --save_dir=${SAVE_DIR}/${TIME}/ &
done
