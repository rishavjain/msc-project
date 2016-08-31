#!/bin/bash

SCRIPT=/home/cop15rj/rishav-msc-project/evaluate/taskdata/scoring/score.pl
GOLD_LST=/home/cop15rj/rishav-msc-project/evaluate/lexsub-master/datasets/lst_all.gold
GOLD_CIC=/home/cop15rj/rishav-msc-project/evaluate/lexsub-master/datasets/coinco_all.no_problematic.gold

win=1
min=100
oper=( mult geomean add avg )
oper=( none )
for op in ${oper[@]}
do
    echo "$win $min $op"
    system="/fastdata/cop15rj/results/scores/wn2/$win/$min/cic/$op/results.generated.best"
    cmd="perl $SCRIPT $system $GOLD_CIC -t best"
    echo $cmd
    eval $cmd

    system="/fastdata/cop15rj/results/scores/wn2/$win/$min/cic/$op/results.generated.oot"
    cmd="perl $SCRIPT $system $GOLD_CIC -t oot"
    echo $cmd
    eval $cmd

    echo
done



