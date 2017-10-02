#!/bin/bash

source venv/bin/activate
COMMAND="vw --quiet -b 24 -d train_glove.vw -c --ftrl -f model.glove --holdout_off --loss_function logistic --link=logistic --passes 11 -q xx"
echo "$COMMAND"
$COMMAND
vw -i model.glove -t -d test_glove.vw -p pred.txt --quiet
python accuracy.py pred.txt testdata.manual.2009.06.14.csv
