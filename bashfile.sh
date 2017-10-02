#!/bin/bash

## Default values, HyperParameter to be tuned
# LEARNING_RATE_DEFAULT = 1e-2
# BATCH_SIZE_DEFAULT = 1
# RANDOM_BATCH_DEFAULT = True
# S_FOLD_DEFAULT = 10
# REPETITION_SCALAR_DEFAULT = 1
# EVAL_FREQ_DEFAULT = S_FOLD_DEFAULT - 1
# CHECKPOINT_FREQ_DEFAULT = MAX_STEPS_DEFAULT
# PRINT_FREQ_DEFAULT = int(MAX_STEPS_DEFAULT/8)
# OPTIMIZER_DEFAULT = 'ADAM'
# WEIGHT_REGULARIZER_DEFAULT = 'l2'
# WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.18
# ACTIVATION_FCT_DEFAULT = 'elu'
# LSTM_SIZE_DEFAULT = 10
# FEAT_EPOCH_DEFAULT = CHECKPOINT_FREQ_DEFAULT-1
# LSTM_SIZE_DEFAULT = 10

### Task: LSTM-Network
## Training
# Examples
# python NeVRo.py --max_steps 15 --log_dir ./LSTM/logs/const_var --checkpoint_dir ./LSTM/checkpoints/lstm/const_var --weight_reg l2 --weight_reg_strength 0.18 --
# python NeVRo.py --learning_rate 1e-4 --checkpoint_dir ./checkpoints/lstm/l1 --weight_reg l1 --weight_reg_strength 0.18
# python NeVRo.py --learning_rate 1e-4 --checkpoint_dir ./checkpoints/lstm/l2 --weight_reg l2 --weight_reg_strength 0.18

# python NeVRo.py --repet_scalar 1000

# Testing
python NeVRo.py --summaries False





