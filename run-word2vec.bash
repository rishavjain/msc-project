#!/bin/bash

tSTART=$(date +%s)

CODE=/home/cop15rj/final
WORD2VEC=/home/cop15rj/final/word2vecf

COUNTFILTER=$WORD2VEC/count_and_filter
WORD2VEC=$WORD2VEC/word2vecf
VEC2NPS=$CODE/vecs2nps.py

INPUT=$1
MODEL_OUT=$2
MIN_COUNT=$3
W2V_NEG=$4 #15
W2V_SIZE=$5 #600

echo "INPUT=$1
MODEL_OUT=$2
MIN_COUNT=$3
W2V_NEG=$4 #15
W2V_SIZE=$5"


# User specific aliases and functions
module load apps/python/anaconda3-2.5.0
module load apps/java/1.8u71
export PYTHONPATH='/home/cop15rj/rishav-msc-project/*:'
export PYTHONIOENCODING=iso-8859-1
CPU=64 #$(grep -c ^processor /proc/cpuinfo)
CURR=$(pwd)
TMP=/scratch/cop15rj/$(date +%s)

mkdir -p $MODEL_OUT
mkdir -p $TMP

function runcommand {
    STARTTIME=$(date +%s)
    cmd=${@:2}

    echo; echo "STARTTIME=$(date)"; echo $cmd
    eval $cmd
    ENDTIME=$(date +%s)
    echo "$(date): It took $(($ENDTIME - $STARTTIME)) seconds to complete '${@:1:1}' task."; echo;
}

if [ $COUNTFILTER ]
then
    runcommand count-filter "$COUNTFILTER \
                    -train $INPUT \
                    -cvocab $MODEL_OUT/cv \
                    -wvocab $MODEL_OUT/wv \
                    -min-count $MIN_COUNT"

#    runcommand copy-cv "cp $TMP/cv $MODEL_OUT/cv"
#    runcommand copy-wv "cp $TMP/wv $MODEL_OUT/wv"
fi

##############################################################################
#runcommand initial-copy-cv "cp $DATA/lexsub/2/cv $TMP/cv"
#runcommand initial-copy-wv "cp $DATA/lexsub/2/wv $TMP/wv"
#runcommand initial-copy-data100 "cp $DATA/lexsub/2/data100.dep $TMP/data100.dep"
##############################################################################

if [ $WORD2VEC ]
then
    runcommand word2vecf "$WORD2VEC \
                            -train $INPUT \
                            -cvocab $MODEL_OUT/cv \
                            -wvocab $MODEL_OUT/wv \
                            -output $MODEL_OUT/dim600vecs \
                            -dumpcv $MODEL_OUT/dim600contexts \
                            -size $W2V_SIZE -negative $W2V_NEG -threads 16"

#    runcommand copy-dim600vecs "cp $TMP/dim600vecs $MODEL_OUT/dim600vecs"
#    runcommand copy-dim600contexts "cp $TMP/dim600contexts $MODEL_OUT/dim600contexts"
fi

if [ $VEC2NPS ]
then
    runcommand vec2nps-wv "python $VEC2NPS $MODEL_OUT/dim600vecs $MODEL_OUT/vecs"
    runcommand vec2nps-cv "python $VEC2NPS $MODEL_OUT/dim600contexts $MODEL_OUT/contexts"

#    runcommand copy-vecs "cp $TMP/vecs* $MODEL_OUT/"
#    runcommand copy-contexts "cp $TMP/contexts* $MODEL_OUT/"
fi

rm -r /scratch/cop15rj/*

tEND=$(date +%s)
echo
echo Finished: total $(($tEND - $tSTART)) seconds
