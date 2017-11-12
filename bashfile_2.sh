#!/bin/bash

# Default values, HyperParameter to be tuned
# LEARNING_RATE_DEFAULT = 1e-2
# BATCH_SIZE_DEFAULT = 1
# RANDOM_BATCH_DEFAULT = True
# S_FOLD_DEFAULT = 10
# REPETITION_SCALAR_DEFAULT = 1
# MAX_STEPS_DEFAULT = REPETITION_SCALAR_DEFAULT*(270 - 270/S_FOLD_DEFAULT)/BATCH_SIZE_DEFAULT
# EVAL_FREQ_DEFAULT = EVAL_FREQ_DEFAULT = (S_FOLD_DEFAULT - 1)/BATCH_SIZE_DEFAULT
# CHECKPOINT_FREQ_DEFAULT = MAX_STEPS_DEFAULT
# PRINT_FREQ_DEFAULT = int(MAX_STEPS_DEFAULT/8)
# OPTIMIZER_DEFAULT = 'ADAM'
# WEIGHT_REGULARIZER_DEFAULT = 'l2'
# WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.18
# ACTIVATION_FCT_DEFAULT = 'elu'
# LSTM_SIZE_DEFAULT = '50, 10'
# FC_NUM_HIDDEN_UNITS = '100'  # this are two layers n_hidden(FC1)=[lstm,100], n_hidden(FC2)=[100,1]
# FEAT_EPOCH_DEFAULT = CHECKPOINT_FREQ_DEFAULT-1
# HILBERT_POWER_INPUT_DEFAULT = True
# COMPONENT_DEFAULT = "best"
# SUMMARIES_DEFAULT = True
# SUBJECT_DEFAULT = 36

# Scripts will be processed successively
### Task: LSTM-Network

## Experiment: Test best x-corr component vs. shuffled values (i.e. noise) vs. random component

# Per Subject: 44(lag=0), 02(lag=1), 22, 28, 05(lag=2), 06, 14, 17, 09(lag=3), 24, 26, 41, 31(lag=4), 43, 04, 07(lag=5), 11, 19, 36
python3 NeVRo.py --subject 2 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
python3 NeVRo.py --subject 2 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
python3 NeVRo.py --subject 2 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/
# python3 NeVRo.py --subject 5 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
# python3 NeVRo.py --subject 5 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
# python3 NeVRo.py --subject 5 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/
# python3 NeVRo.py --subject 14 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
# python3 NeVRo.py --subject 14 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
# python3 NeVRo.py --subject 14 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/
# python3 NeVRo.py --subject 9 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
# python3 NeVRo.py --subject 9 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
# python3 NeVRo.py --subject 9 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/
# python3 NeVRo.py --subject 26 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
# python3 NeVRo.py --subject 26 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
# python3 NeVRo.py --subject 26 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/
# python3 NeVRo.py --subject 31 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
# python3 NeVRo.py --subject 31 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
# python3 NeVRo.py --subject 31 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/
# python3 NeVRo.py --subject 4 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
# python3 NeVRo.py --subject 4 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
# python3 NeVRo.py --subject 4 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/
# python3 NeVRo.py --subject 11 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
# python3 NeVRo.py --subject 11 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
# python3 NeVRo.py --subject 11 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/

# python3 NeVRo.py --subject 36 --repet_scalar 750 --component best --lstm_size 50,50 --path_specificities best/
# python3 NeVRo.py --subject 36 --repet_scalar 750 --component noise --lstm_size 50,50 --path_specificities noise/
# python3 NeVRo.py --subject 36 --repet_scalar 750 --component random --lstm_size 50,50 --path_specificities random/
