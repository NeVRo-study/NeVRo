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

# Testing
# python3 NeVRo.py --summaries False
# python3 LSTM_pred_plot.py Save_plots Path_Specificities(empty or 'subfolder/')
# python3 LSTM_pred_plot.py True lstm-150_fc-150/
# python3 NeVRo.py --subject 36 --lstm_size 150,50 --summaries True --plot True --path_specificities test_lstm-150-50/
# python3 NeVRo.py --subject 36 --lstm_size 150 --fc_n_hidden 100 --summaries False --plot True --path_specificities test2_lstm-150_fc-100/
# python3 NeVRo.py --subject 36 --lstm_size 150,100 --fc_n_hidden 100 --summaries False --plot True --path_specificities test2_lstm-150-100_fc-150/

## Training

# Learning Rate
python3 NeVRo.py --subject 36 --learning_rate 1e-2 --summaries False --plot True --path_specificities lr-1e-2/
python3 NeVRo.py --subject 36 --learning_rate 1e-3 --summaries False --plot True --path_specificities lr-1e-3/
python3 NeVRo.py --subject 36 --learning_rate 1e-4 --summaries False --plot True --path_specificities lr-1e-4/

# Weight Reg
python3 NeVRo.py --subject 36 --weight_reg none --summaries False --plot True --path_specificities wreg-none/

python3 NeVRo.py --subject 36 --weight_reg l1 --weight_reg_strength 0.18 --plot True --path_specificities wreg-l1-018/
python3 NeVRo.py --subject 36 --weight_reg l1 --weight_reg_strength 0.36 --plot True --path_specificities wreg-l1-036/
python3 NeVRo.py --subject 36 --weight_reg l1 --weight_reg_strength 0.72 --plot True --path_specificities wreg-l1-072/

python3 NeVRo.py --subject 36 --weight_reg l2 --weight_reg_strength 0.18 --plot True --path_specificities wreg-l2-018/
python3 NeVRo.py --subject 36 --weight_reg l2 --weight_reg_strength 0.36 --plot True --path_specificities wreg-l2-036/
python3 NeVRo.py --subject 36 --weight_reg l2 --weight_reg_strength 0.72 --plot True --path_specificities wreg-l2-072/

# Activation Function
python3 NeVRo.py --subject 36 --activation_fct elu --plot True --path_specificities actfunc-elu/
python3 NeVRo.py --subject 36 --activation_fct relu --plot True --path_specificities actfunc-relu/

# LSTM-Size
# 1-LSTM + 1-FC
python3 NeVRo.py --subject 36 --lstm_size 50 --plot True --path_specificities lstm-50/
python3 NeVRo.py --subject 36 --lstm_size 100 --plot True --path_specificities lstm-100/
python3 NeVRo.py --subject 36 --lstm_size 200 --plot True --path_specificities lstm-200/
python3 NeVRo.py --subject 36 --lstm_size 250 --plot True --path_specificities lstm-250/

# 2-LSTM + 1-FC
python3 NeVRo.py --subject 36 --lstm_size 50,50 --plot True --path_specificities lstm-50-50/
python3 NeVRo.py --subject 36 --lstm_size 100,50 --plot True --path_specificities lstm-100-50/
python3 NeVRo.py --subject 36 --lstm_size 200,50 --plot True --path_specificities lstm-200-50/
python3 NeVRo.py --subject 36 --lstm_size 250,50 --plot True --path_specificities lstm-250-50/

# 1-LSTM + 2-FC
python3 NeVRo.py --subject 36 --fc_n_hidden 25 --plot True --path_specificities fc-25/
python3 NeVRo.py --subject 36 --fc_n_hidden 50 --plot True --path_specificities fc-50/
python3 NeVRo.py --subject 36 --fc_n_hidden 100 --plot True --path_specificities fc-100/
python3 NeVRo.py --subject 36 --fc_n_hidden 200 --plot True --path_specificities fc-200/

# 2-LSTM + 2-FC
python3 NeVRo.py --subject 36 --lstm_size 50,50 --fc_n_hidden 25 --plot True --path_specificities lstm-50-50_fc-25/
python3 NeVRo.py --subject 36 --lstm_size 100,50 --fc_n_hidden 50 --plot True --path_specificities lstm-100-50_fc-50/
python3 NeVRo.py --subject 36 --lstm_size 200,50 --fc_n_hidden 100 --plot True --path_specificities lstm-200-50_fc-100/
python3 NeVRo.py --subject 36 --lstm_size 250,50 --fc_n_hidden 200 --plot True --path_specificities lstm-250-50_fc-200/