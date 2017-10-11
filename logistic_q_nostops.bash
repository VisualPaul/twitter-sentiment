#!/bin/bash

source venv/bin/activate
COMMAND="vw --quiet -b 24 -d train_naive_nostops.vw -c --ftrl --l1 1e4 --l2 1e-4 -f model.naive_nostops --holdout_off --loss_function logistic --link=logistic --passes 2500 --ngram x2"
echo "$COMMAND"
$COMMAND
vw -i model.naive_nostops -t -d test_naive_nostops.vw -p pred_naive_nostops.txt --quiet
python accuracy.py pred_naive_nostops.txt testdata.manual.2009.06.14.csv
