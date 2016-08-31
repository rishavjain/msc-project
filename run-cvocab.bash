#!/bin/bash

tSTART=$(date +%s)

CODE=/home/cop15rj/final
OUT=/fastdata/cop15rj/out/contexts
DATA=/data/cop15rj/data100.conll.gz

WORD2VEC=$CODE/word2vecf
EXTRACTDEPS=$CODE/extract-deps.py
COUNTFILTER=$WORD2VEC/count_and_filter
FILTERCONTEXTS=$CODE/filter-contexts.py

MIN_COUNT=$1
WIN_SIZE=$2

# User specific aliases and functions
module load apps/python/anaconda3-2.5.0
module load apps/java/1.8u71
export PYTHONPATH='/home/cop15rj/rishav-msc-project/*:'
export PYTHONIOENCODING=iso-8859-1
CURR=$(pwd)
TMP=$OUT #/scratch/cop15rj/$(date +%s)

function runcommand {
    STARTTIME=$(date +%s)
    cmd=${@:2}

    echo; echo "STARTTIME=$(date)"; echo $cmd
    eval $cmd
    ENDTIME=$(date +%s)
    echo "$(date): It took $(($ENDTIME - $STARTTIME)) seconds to complete '${@:1:1}' task."; echo;
}

echo "CODE=/home/cop15rj/final
OUT=/fastdata/cop15rj/out/contexts
WORD2VEC=/home/cop15rj/final/word2vecf

EXTRACTDEPS=$CODE/extract-deps.py
COUNTFILTER=$WORD2VEC/count_and_filter
FILTERCONTEXTS=$CODE/filter-contexts.py

MIN_COUNT=$1
WIN_SIZE=$2"

mkdir -p $OUT
mkdir -p $TMP


#DATA=/home/cop15rj/final/in/ukwac_1M.conll.gz

if [ $EXTRACTDEPS ]
then
    runcommand extract-deps "zcat $DATA | python $EXTRACTDEPS \
                                            /fastdata/cop15rj/out/vocab-all.txt 0 $WIN_SIZE > \
                                            $TMP/deps-$MIN_COUNT-$WIN_SIZE.dep"

    runcommand copy-deps1 "cp $TMP/deps-$MIN_COUNT-$WIN_SIZE.dep $OUT/deps-$MIN_COUNT-$WIN_SIZE.dep0"
fi

if [ $COUNTFILTER ]
then
    runcommand count-filter "$COUNTFILTER \
                    -train $TMP/deps-$MIN_COUNT-$WIN_SIZE.dep \
                    -cvocab $TMP/cv-$MIN_COUNT-$WIN_SIZE \
                    -wvocab $TMP/wv-$MIN_COUNT-$WIN_SIZE \
                    -min-count 0"

    runcommand copy-cv "cp $TMP/cv-$MIN_COUNT-$WIN_SIZE $OUT/cv-$MIN_COUNT-$WIN_SIZE"
fi

if [ $FILTERCONTEXTS ]
then
    runcommand filter-contexts "python $FILTERCONTEXTS \
                                    /fastdata/cop15rj/out/vocab-all.txt \
                                    $OUT/cv-$MIN_COUNT-$WIN_SIZE \
                                    $MIN_COUNT \
                                    $WIN_SIZE \
                                    $DATA \
                                    $TMP/deps-$MIN_COUNT-$WIN_SIZE.dep"

    runcommand copy-deps "cp $TMP/deps-$MIN_COUNT-$WIN_SIZE.dep $OUT/deps-$MIN_COUNT-$WIN_SIZE.dep"
fi


rm -r /scratch/cop15rj/*

tEND=$(date +%s)
echo
echo Finished: total $(($tEND - $tSTART)) seconds
