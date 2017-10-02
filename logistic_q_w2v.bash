#!/bin/bash

source venv/bin/activate
COMMAND="vw --quiet -b 24 -d train_w2v.vw -c --ftrl -f model_w2v_q --holdout_off --loss_function logistic --link=logistic --passes 10 -q xx"
echo "$COMMAND"
$COMMAND
vw -i model_w2v_q -t -d test_w2v.vw -p pred_w2v_q.txt --quiet
python accuracy.py pred_w2v_q.txt testdata.manual.2009.06.14.csv
