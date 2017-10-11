#!/bin/bash

## Default values, HyperParameter to be tuned
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
# Examples
# python NeVRo.py --max_steps 15 --log_dir ./LSTM/logs/const_var --checkpoint_dir ./LSTM/checkpoints/lstm/const_var --weight_reg l2 --weight_reg_strength 0.18 --
# python NeVRo.py --learning_rate 1e-4 --checkpoint_dir ./checkpoints/lstm/l1 --weight_reg l1 --weight_reg_strength 0.18
# python NeVRo.py --learning_rate 1e-4 --checkpoint_dir ./checkpoints/lstm/l2 --weight_reg l2 --weight_reg_strength 0.18

# python3 NeVRo.py --repet_scalar 1000

# Testing
# python3 NeVRo.py --summaries False
# python3 LSTM_pred_plot.py Debug Save_plots Path_Specificities(empty or 'subfolder/')
# python3 LSTM_pred_plot.py False False
# python3 LSTM_pred_plot.py True True lstm-150_fc-150/

# python3 NeVRo.py --subject 36 --lstm_size 150,50 --summaries False --plot True --path_specificities lstm-150-50/
# python3 NeVRo.py --subject 36 --lstm_size 150 --fc_n_hidden 150 --summaries False --plot True --path_specificities lstm-150_fc-150/
python3 NeVRo.py --subject 36 --lstm_size 150,100 --summaries False --plot False --path_specificities lstm-150-100/




