#!/bin/bash

source venv/bin/activate
COMMAND="vw --quiet -b 24 -d train.vw -c --ftrl -f model --holdout_off --loss_function logistic --link=logistic --passes 10"
echo "$COMMAND"
$COMMAND
vw -i model -t -d test.vw -p pred.txt --quiet
python accuracy.py pred.txt testdata.manual.2009.06.14.csv
