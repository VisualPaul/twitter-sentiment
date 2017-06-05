#!/bin/bash

python2 to_vw.py training.1600000.processed.noemoticon.csv train_w2v.vw testdata.manual.2009.06.14.csv test_w2v.vw -p -v w2v $@
rm *.cache
