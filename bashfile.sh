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
# SUMMARIES_DEFAULT = True
# SUBJECT_DEFAULT = 36

# Scripts will be processed successively
### Task: LSTM-Network

## Training

# Per Subject
python3 NeVRo.py --subject 44
python3 NeVRo.py --subject 2
python3 NeVRo.py --subject 22
python3 NeVRo.py --subject 28
python3 NeVRo.py --subject 5
python3 NeVRo.py --subject 6
python3 NeVRo.py --subject 14
python3 NeVRo.py --subject 17
python3 NeVRo.py --subject 9
python3 NeVRo.py --subject 24
# python3 NeVRo.py --subject 26
# python3 NeVRo.py --subject 41
# python3 NeVRo.py --subject 31
# python3 NeVRo.py --subject 43
# python3 NeVRo.py --subject 4
# python3 NeVRo.py --subject 7
# python3 NeVRo.py --subject 11
# python3 NeVRo.py --subject 19

# python3 NeVRo.py --subject 36
