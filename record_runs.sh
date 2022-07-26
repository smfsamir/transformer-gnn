#!/bin/bash
BS=$1
echo $BS
FNAME="results/batch-${BS}-fmha.csv"
printf 'test_acc,val_loss,best_epoch' >| FNAME
for i in `seq 1 20`;
do
    OUTPUT=$(python main_cora_gat.py $BS |tail -1)
    echo "${OUTPUT}" >> $FNAME
done