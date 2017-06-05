#!/bin/bash

python2 to_vw.py training.1600000.processed.noemoticon.csv train_tfidf.vw testdata.manual.2009.06.14.csv test_tfidf.vw -p $@
rm *.cache
