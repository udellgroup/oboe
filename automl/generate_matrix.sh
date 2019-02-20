#!/usr/bin/env bash

# Shell script to generate error matrix (and merge results), parallelized across datasets.

usage () {
cat <<HELP_USAGE
Usage:
$0  [-m] mode [-s] SAVE_DIR [-d] DATA_DIR [-p] P_TYPE [-j] JSON_FILE [-e] ERROR_MATRIX [-n] MAX_PROCS [-a] AUC [-f] FULLNAME [-h]

-m:         mode in which to run, either "generate" or "merge"
-s:         where to save results, or in the merge mode, where results are saved
-d:         path to directory containing training datasets are located
-p:         problem type, either "classification" or "regression"
-j:         path to model configurations json file
-e:         error matrix already generated
-n:         maximum number of processes assigned to error matrix generation
-a:         whether to use AUC instead of BER
-f:         whether to use dataset file full name as dataset name
-h:         show this help information
HELP_USAGE
}

# parse user arguments
while getopts ":hm:s:d:p:j:e:n:a:f:" opt; do
    case ${opt} in
    h)
        usage
        exit 1
        ;;
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
    s)
        SAVE_DIR=$OPTARG
        ;;
    d)
        DATA_DIR=$OPTARG
        ;;
    p)
        P_TYPE=$OPTARG
        ;;
    j)
        JSON_FILE=$OPTARG
        ;;
    e)
        ERROR_MATRIX=$OPTARG
        ;;
    n)
        MAX_PROCS=$OPTARG
        ;;
    a)
        AUC=$OPTARG
        ;;
    f)
        FULLNAME=$OPTARG
        ;;
    \?)
        echo "Invalid option: -${OPTARG}" >&2
        usage
        exit 1
        ;;
esac
done

#if [ "$1" == "" ]
#then
#  echo "Must specify mode."
#  usage
#  exit 1
#fi

# no limit for maximum number of processes if no number is given
if [ "${MAX_PROCS}" == "" ]
then
    MAX_PROCS="0"
fi

# default to not using AUC
if [ "${AUC}" == "" ]
then
    AUC="False"
fi

# default to not using fullname
if [ "${FULLNAME}" == "" ]
then
    FULLNAME="False"
fi

# strip '/' from end of file path (if there is one)
#SAVE_DIR=${3%/}
#DATA_DIR=${4%/}

# location of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# generate mode
if [ "${mode}" == "generate" ]
then
  time=`date +%Y%m%d%H%M`
  mkdir -p ${SAVE_DIR}/${time}
  echo -e "SAVE_DIR=${SAVE_DIR}\nDATA_DIR=${DATA_DIR}\nP_TYPE=${P_TYPE}\nJSON_FILE=${JSON_FILE}\nAUC=${AUC}\nERROR_MATRIX=${ERROR_MATRIX}\n" >> ${SAVE_DIR}/${time}/configurations.txt
  echo "Error matrix generation started at ${time}" >> ${SAVE_DIR}/${time}/log_${time}.txt

  ls ${DATA_DIR}/*.csv | xargs -i --max-procs=${MAX_PROCS} bash -c \
  "python ${DIR}/generate_vector.py '${P_TYPE}' {} --file=${JSON_FILE} --save_dir=${SAVE_DIR}/${time} \
  --error_matrix=${ERROR_MATRIX} --auc=${AUC} --fullname=${FULLNAME} &>> ${SAVE_DIR}/${time}/warnings_and_errors.txt"
fi

# merge mode
if [ "${mode}" == "merge" ]
then
  python ${DIR}/util.py ${SAVE_DIR}
fi
