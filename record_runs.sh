#!/bin/bash
BS=$1
NUM_SG=$2
echo $BS
echo $NUM_SG
FNAME="results/batch-${BS}_num_sg-${NUM_SG}.csv"
printf 'test_acc,val_loss,best_epoch' >| FNAME
for i in `seq 1 20`;
do
    OUTPUT=$(python main_cora_gat.py $BS $NUM_SG |tail -1)
    echo "${OUTPUT}" >> $FNAME
done