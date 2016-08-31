#!/bin/bash

tSTART=$(date +%s)

CODE=/home/cop15rj/final

EVALUATE=$CODE/evaluate.py
GAP=$CODE/old/gap.py

#INPUT_EMB=$1
INPUT_DST=$1
OUT=$2
EVALOP=$3 #mult, add, avg, geomean
DATASET=$4 #lst, cic
WIN_SIZE=$5


INPUT_EMB=$CODE/testdata

# User specific aliases and functions
module load apps/python/anaconda3-2.5.0
module load apps/java/1.8u71
export PYTHONPATH='/home/cop15rj/rishav-msc-project/*:'
export PYTHONIOENCODING=iso-8859-1
CPU=64 #$(grep -c ^processor /proc/cpuinfo)
CURR=$(pwd)
TMP=/scratch/cop15rj/$(date +%s)

function runcommand {
    STARTTIME=$(date +%s)
    cmd=${@:2}

    echo; echo "STARTTIME=$(date)"; echo $cmd
    eval $cmd
    ENDTIME=$(date +%s)
    echo "$(date): It took $(($ENDTIME - $STARTTIME)) seconds to complete '${@:1:1}' task."; echo;
}

echo "INPUT_EMB=$1
INPUT_DST=$2
OUT=$3
EVALOP=$4
DATASET=$5
WIN_SIZE=$6"

mkdir -p $OUT

###############################################################################
#runcommand install-nltk "pip install --user nltk"
#runcommand nltk-download "python -m nltk.downloader all"
###############################################################################

if [ $EVALUATE ]
then
    runcommand evaluate "python $EVALUATE $EVALOP $INPUT_EMB $INPUT_DST $OUT $DATASET $WIN_SIZE"
    runcommand gap "python $GAP $INPUT_DST $OUT $DATASET"
fi

tEND=$(date +%s)
echo
echo Finished: total $(($tEND - $tSTART)) seconds
