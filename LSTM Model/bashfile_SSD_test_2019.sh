#!/usr/bin/env bash

# 2019 SSD Test Bashfile:
python3 NeVRo.py --subject 36 --seed True --task regression --repet_scalar 2 --lstm_size 10 --fc_n_hidden 1 --component 1,3,5 --condition nomov
