#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install --user
pip install -e .

# EMD
cd $HOME/extensions/emd
pip install -e .