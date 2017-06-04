#!/bin/sh

curl -o data.zip http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip data.zip
curl https://nlp.stanford.edu/data/glove.twitter.27B.zip | bsdtar -xvf- glove.twitter.27B.200d.txt
