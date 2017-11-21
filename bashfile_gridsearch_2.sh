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

## Hyperparameter Search (HPS)

# Specific Search: 2-LSTM + 1-FC, Learning Rate, Weight Reg-Strength
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 75,25 --path_specificities HPS_lstm-75-25/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 75,25 --learning_rate 1e-4 --path_specificities HPS_lstm-75-25_lr-1e-4/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 75,25 --weight_reg_strength 0.36 --path_specificities HPS_lstm-75-25_l2-036/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 75,25 --weight_reg_strength 0.36 --learning_rate 1e-4 --path_specificities HPS_lstm-75-25_l2-036_lr-1e-4/

# Train with 3 components (>1) non-band-passed
# python3 NeVRo.py --subject 36 --seed True --band_pass False --component 1,2,3,4,5 --repet_scalar 320 --successive 3 --lstm_size 50,30 --learning_rate 1e-4 --path_specificities HPS_lstm-50-30_lr-1e-4_comp-1-2-3-4-5_bpass-F_suc-3/
# python3 NeVRo.py --subject 36 --seed True --band_pass False --component 1,2,3,4,5 --repet_scalar 320 --lstm_size 40,20 --path_specificities HPS_lstm-40-20_comp-1-2-3-4-5_bpass-T/


