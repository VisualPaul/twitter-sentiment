#!/bin/bash

python2 to_vw.py training.1600000.processed.noemoticon.csv train.vw.new testdata.manual.2009.06.14.csv test.vw -p
shuf train.vw.new > train.vw
rm train.vw.new *.cache
