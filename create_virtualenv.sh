#!/bin/bash
echo "Prerequesitest: python, pip, virtualenv + blas, gcc, etc for the numpy"
virtualenv venv
source venv/bin/activate
pip install numpy pandas scipy scikit-learn gensim
