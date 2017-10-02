#!/bin/bash

source venv/bin/activate
COMMAND="vw --quiet -b 24 -d train_w2v.vw -c --ftrl -f model_w2v --holdout_off --loss_function logistic --link=logistic --passes 1"
echo "$COMMAND"
$COMMAND
vw -i model_w2v -t -d test_w2v.vw -p pred_w2v.txt --quiet
python accuracy.py pred_w2v.txt testdata.manual.2009.06.14.csv
