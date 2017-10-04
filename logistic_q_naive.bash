#!/bin/bash

source venv/bin/activate
COMMAND="vw --quiet -b 24 -d train_naive.vw -c --ftrl --l1 1e4 --l2 1e-4 -f model.naive --holdout_off --loss_function logistic --link=logistic --passes 2500 --ngram x2"
echo "$COMMAND"
$COMMAND
vw -i model.naive -t -d test_naive.vw -p pred_naive.txt --quiet
python accuracy.py pred_naive.txt testdata.manual.2009.06.14.csv
