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

## Test model's learning ability with inverse+stretch+noise ratings as input
# python3 NeVRo.py --testmodel True --subject 36 --seed True --repet_scalar 320 --lstm_size 50,30 --path_specificities TESTMODE_lstm-50-30/
# python3 NeVRo.py --testmodel True --subject 36 --seed True --repet_scalar 80 --lstm_size 50,30 --path_specificities TESTMODE-exp3_lstm-50-30/
# python3 NeVRo.py --testmodel True --subject 36 --seed True --repet_scalar 80 --lstm_size 50,30 --path_specificities TESTMODE-global-slope_lstm-50-30/
# python3 NeVRo.py --testmodel True --subject 36 --seed True --repet_scalar 80 --lstm_size 50,30 --path_specificities TESTMODE-local-slope_lstm-50-30/
# python3 NeVRo.py --testmodel True --subject 36 --seed True --repet_scalar 80 --lstm_size 50,30 --path_specificities TESTMODE-strong-local-slope_lstm-50-30/
# python3 NeVRo.py --testmodel True --subject 36 --seed True --repet_scalar 80 --lstm_size 50,30 --path_specificities TESTMODE-inverse-local-slope_lstm-50-30/
# python3 NeVRo.py --testmodel True --subject 2 --seed True --repet_scalar 80 --lstm_size 40,20 --path_specificities TESTMODE-neg-strong-local-slope_lstm-40-20/


## Hyperparameter Search (HPS)

# Specific Search: 2-LSTM + 1-FC, Learning Rate, Weight Reg-Strength
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 50,25 --path_specificities HPS_lstm-50-25/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 50,25 --learning_rate 1e-4 --path_specificities HPS_lstm-50-25_lr-1e-4/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 50,25 --weight_reg_strength 0.36 --path_specificities HPS_lstm-50-25_l2-036/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 50,25 --weight_reg_strength 0.36 --learning_rate 1e-4 --path_specificities HPS_lstm-50-25_l2-036_lr-1e-4/

# Specific Search: 2-LSTM + 1-FC, Successive Batches
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --successive 3 --lstm_size 50,25 --path_specificities HPS_lstm-50-25_suc-3/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --successive 3 --lstm_size 50,40 --learning_rate 1e-2 --path_specificities HPS_lstm-50-40_lr-1e-2_suc-3/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --successive 3 --lstm_size 50,40 --learning_rate 1e-4 --path_specificities HPS_lstm-50-40_lr-1e-4_suc-3/

# Best so far
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --lstm_size 100 --weight_reg_strength 0.36 --path_specificities HPS_lstm-100_l2-0.36/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --successive 3 --lstm_size 50,40 --learning_rate 1e-4 --path_specificities HPS_lstm-50-40_lr-1e-4_suc-3/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --successive 3 --lstm_size 50,30 --learning_rate 1e-4 --path_specificities HPS_lstm-50-30_lr-1e-4_suc-3/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --successive 3 --lstm_size 50,40 --path_specificities HPS_lstm-50-40_suc-3/
# python3 NeVRo.py --subject 36 --seed True --repet_scalar 320 --successive 3 --lstm_size 50,40 --weight_reg_strength 0.09 --path_specificities HPS_lstm-50-40_l2-009_suc-3/

# No Hilbert
# python3 NeVRo.py --subject 36 --seed True --hilbert_power False --repet_scalar 320 --successive 3 --lstm_size 50,40 --learning_rate 1e-4 --path_specificities HPS_lstm-50-40_lr-1e-4_hilb-F_suc-3/
# python3 NeVRo.py --subject 36 --seed True --hilbert_power False --repet_scalar 320 --successive 3 --lstm_size 50,30 --learning_rate 1e-4 --path_specificities HPS_lstm-50-30_lr-1e-4_hilb-F_suc-3/

# Train with (x>1) components
# python3 NeVRo.py --subject 36 --seed True --component 1,3,5 --repet_scalar 320 --successive 3 --lstm_size 50,40 --learning_rate 1e-4 --path_specificities HPS_lstm-50-40_lr-1e-4_comp-1-3-5_suc-3/
# python3 NeVRo.py --subject 36 --seed True --component 1,3,5 --repet_scalar 320 --successive 3 --lstm_size 50,30 --learning_rate 1e-4 --path_specificities HPS_lstm-50-30_lr-1e-4_comp-1-3-5_suc-3/
# python3 NeVRo.py --subject 36 --seed True --component 1,2,3,4,5 --repet_scalar 320 --successive 3 --lstm_size 50,30 --learning_rate 1e-4 --path_specificities HPS_lstm-50-30_lr-1e-4_comp-1-2-3-4-5_suc-3/

# Train SPOC
# python3 NeVRo.py --subject 36 --seed True --filetype SPOC --repet_scalar 320 --successive 3 --lstm_size 50,40 --learning_rate 1e-4 --path_specificities HPS_lstm-50-40_lr-1e-4_ftype-spoc_suc-3/
# python3 NeVRo.py --subject 36 --seed True --filetype SPOC --repet_scalar 320 --successive 3 --lstm_size 50,30 --learning_rate 1e-4 --path_specificities HPS_lstm-50-30_lr-1e-4_ftype-spoc_suc-3/

# Train non-band-passed
# python3 NeVRo.py --subject 36 --seed True --band_pass False --repet_scalar 320 --successive 3 --lstm_size 50,40 --learning_rate 1e-4 --path_specificities HPS_lstm-50-40_lr-1e-4_bpass-F_suc-3/
# python3 NeVRo.py --subject 36 --seed True --band_pass False --repet_scalar 320 --successive 3 --lstm_size 50,30 --learning_rate 1e-4 --path_specificities HPS_lstm-50-30_lr-1e-4_bpass-F_suc-3/

# Train with (x>1) components non-band-passed or SPOC
# python3 NeVRo.py --subject 36 --seed True --band_pass False --component 1,3,5 --repet_scalar 320 --successive 3 --lstm_size 50,40 --learning_rate 1e-4 --path_specificities HPS_lstm-50-40_lr-1e-4_comp-1-3-5_bpass-F_suc-3/
# python3 NeVRo.py --subject 36 --seed True --filetype SPOC --component 1,2,3,4,5,6,7 --repet_scalar 320 --successive 3 --lstm_size 40,30 --path_specificities HPS_lstm-40-30_ftype-spoc_comp-1-2-3-4-5-6-7_suc-3/
# python3 NeVRo.py --subject 2 --seed True --filetype SPOC --component 1,2,3,4,5,6,7 --repet_scalar 320 --successive 3 --lstm_size 40,30 --path_specificities HPS_lstm-40-30_ftype-spoc_comp-1-2-3-4-5-6-7_suc-3/
# python3 NeVRo.py --subject 36 --seed True --filetype SPOC --component 1,2,3,4,5,6,7 --repet_scalar 320 --successive 3 --lstm_size 30,20 --path_specificities HPS_lstm-30-20_ftype-spoc_comp-1-2-3-4-5-6-7_suc-3/
# python3 NeVRo.py --subject 2 --seed True --filetype SPOC --component 1,2,3,4,5,6,7 --repet_scalar 320 --successive 3 --lstm_size 30,20 --path_specificities HPS_lstm-30-20_ftype-spoc_comp-1-2-3-4-5-6-7_suc-3/
# python3 NeVRo.py --subject 36 --seed True --filetype SPOC --component 1,2,3,4,5,6,7 --repet_scalar 320 --successive 3 --lstm_size 20,10 --path_specificities HPS_lstm-20-10_ftype-spoc_comp-1-2-3-4-5-6-7_suc-3/
# python3 NeVRo.py --subject 2 --seed True --filetype SPOC --component 1,2,3,4,5,6,7 --repet_scalar 320 --successive 3 --lstm_size 20,10 --path_specificities HPS_lstm-20-10_ftype-spoc_comp-1-2-3-4-5-6-7_suc-3/
# python3 NeVRo.py --subject 36 --seed True --component 1,2,3,4,5 --repet_scalar 320 --successive 3 --band_pass False --lstm_size 20,10 --weight_reg_strength 0.72 --path_specificities HPS_lstm-20-10_l2-072_comp-1-2-3-4-5_bpass-F_suc-3/
# python3 NeVRo.py --subject 2 --seed True --component 1,2,3,4,5 --repet_scalar 320 --successive 3 --band_pass False --lstm_size 20,10 --weight_reg_strength 0.72 --path_specificities HPS_lstm-20-10_l2-072_comp-1-2-3-4-5_bpass-F_suc-3/
# python3 NeVRo.py --subject 36 --seed True --component 1,2,3,4,5 --repet_scalar 320 --successive 3 --band_pass False --lstm_size 15,10 --weight_reg_strength 0.72 --path_specificities HPS_lstm-15-10_l2-072_comp-1-2-3-4-5_bpass-F_suc-3/
# python3 NeVRo.py --subject 2 --seed True --component 1,2,3,4,5 --repet_scalar 320 --successive 3 --band_pass False --lstm_size 15,10 --weight_reg_strength 0.72 --path_specificities HPS_lstm-15-10_l2-072_comp-1-2-3-4-5_bpass-F_suc-3/
# python3 NeVRo.py --subject 36 --seed True --component 1,2,3,4,5 --repet_scalar 320 --successive 3 --band_pass False --lstm_size 30 --weight_reg_strength 0.72 --path_specificities HPS_lstm-30_l2-072_comp-1-2-3-4-5_bpass-F_suc-3/
# python3 NeVRo.py --subject 2 --seed True --component 1,2,3,4,5 --repet_scalar 320 --successive 3 --band_pass False --lstm_size 30 --weight_reg_strength 0.72 --path_specificities HPS_lstm-30_l2-072_comp-1-2-3-4-5_bpass-F_suc-3/

# Binary Classification
# python3 NeVRo.py --subject 36 --seed True --task classification --shuffle True --repet_scalar 320 --lstm_size 30,10 --path_specificities BiCl_HPS_lstm-30-10/
# python3 NeVRo.py --subject 36 --seed True --task classification --shuffle True --repet_scalar 320 --lstm_size 30,10 --band_pass False --path_specificities BiCl_HPS_lstm-30-10_bpass-F/
# python3 NeVRo.py --subject 36 --seed True --task classification --shuffle True --repet_scalar 320 --lstm_size 20,10 --path_specificities BiCl_HPS_lstm-20-10/
# python3 NeVRo.py --subject 36 --seed True --task classification --shuffle True --repet_scalar 320 --lstm_size 20,10 --band_pass False --path_specificities BiCl_HPS_lstm-20-10_bpass-F/
# python3 NeVRo.py --subject 36 --seed True --task classification --shuffle True --repet_scalar 200 --lstm_size 40,15 --path_specificities BiCl_HPS_lstm-40-15/
# python3 NeVRo.py --subject 36 --seed True --task classification --shuffle True --repet_scalar 200 --lstm_size 40,15 --band_pass False --path_specificities BiCl_HPS_lstm-40-15_bpass-F/

