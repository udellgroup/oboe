#!/usr/bin/env bash

# Shell script to generate error matrix (and merge results), parallelizing across datasets.

usage () {
    cat <<HELP_USAGE
    Usage:
    $0  [-m] mode <max_procs> <save_dir> <data_dir> <p_type> <json>

   -m:         mode in which to run, either "generate" or "merge".
   <save_dir>: (g) where to save results / (m) where results are saved.
   <data_dir>: (g) path to directory containing training datasets are located.
   <p_type>:   (g) problem type, either "classification" or "regression".
   <json>:     (g) path to model configurations json file.
   <err_mtx>   (g) error matrix already generated.
   <max_procs>:(g) maximum number of processes assigned to error matrix generation.

HELP_USAGE
}

# parse user arguments
while getopts ":m:" opt; do
  case ${opt} in
    m)
      if [ ${OPTARG} != "generate" ] && [ ${OPTARG} != "merge" ]
      then
        echo "Invalid mode."
        usage
        exit 1
      fi
      echo "Running in ${OPTARG} mode..." >&2
      mode=${OPTARG}
      ;;
    \?)
      echo "Invalid option: -${OPTARG}" >&2
      usage
      exit 1
      ;;
  esac
done

if [ "$1" == "" ]
then
  echo "Must specify mode."
  usage
  exit 1
fi

# no limit for maximum number of processes if no number is given
if [ "$8" == "" ]
then
  "$8" = "0"
fi

# strip '/' from end of file path (if there is one)
SAVE_DIR=${3%/}
DATA_DIR=${4%/}
P_TYPE=$5
JSON_FILE=$6
ERROR_MATRIX=$7
MAX_PROCS=$8

# location of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# generate mode - runs at most 90 parallel processes (can be changed by editing --max-procs=90 below)
if [ "${mode}" == "generate" ]
then
  time=`date +%Y%m%d%H%M`
  mkdir -p ${SAVE_DIR}/${time}
  echo "Error matrix generation started at ${time}" >> ${SAVE_DIR}/${time}/log_${time}.txt

  ls ${DATA_DIR}/*.csv | xargs -i --max-procs=${MAX_PROCS} bash -c \
  "echo {}; python ${DIR}/generate_vector.py '${P_TYPE}' {} --file=${JSON_FILE} --save_dir=${SAVE_DIR}/${time} --error_matrix=${ERROR_MATRIX}"
fi

# merge mode
if [ "${mode}" == "merge" ]
then
  python ${DIR}/util.py ${SAVE_DIR}
fi
