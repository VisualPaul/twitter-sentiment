#!/bin/bash

source venv/bin/activate
COMMAND="vw --quiet -b 24 -d train_tfidf.vw -c --ftrl --l1 1e4 --l2 1e-4 -f model.tfidf --holdout_off --loss_function logistic --link=logistic --passes 150"
echo "$COMMAND"
$COMMAND
vw -i model.tfidf -t -d test_tfidf.vw -p pred.txt --quiet
python accuracy.py pred.txt testdata.manual.2009.06.14.csv
