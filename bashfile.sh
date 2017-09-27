#!/bin/bash

## Default values, HyperParameter to be tuned
# LEARNING_RATE_DEFAULT = 1e-2
# BATCH_SIZE_DEFAULT = 1  # or bigger
# S_FOLD_DEFAULT = 10
# MAX_STEPS_DEFAULT = 15
# EVAL_FREQ_DEFAULT = MAX_STEPS_DEFAULT/15
# CHECKPOINT_FREQ_DEFAULT = MAX_STEPS_DEFAULT/3
# PRINT_FREQ_DEFAULT = 5
# OPTIMIZER_DEFAULT = 'ADAM'
# WEIGHT_REGULARIZER_DEFAULT = 'l2'
# WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.18
# ACTIVATION_FCT_DEFAULT = 'elu'
# LSTM_SIZE_DEFAULT = 10

### Task: LSTM-Network
## Training
# python3 NeVRo.py --max_steps 15 --log_dir ./LSTM/logs/const_var --checkpoint_dir ./LSTM/checkpoints/lstm/const_var --weight_reg l2 --weight_reg_strength 0.18 --
# python3 NeVRo.py --max_steps 15
python NeVRo.py --repet_scalar 1000

# python2 NeVRo.py --learning_rate 1e-4 --checkpoint_dir ./checkpoints/lstm/l1 --weight_reg l1 --weight_reg_strength 0.18
# python2 NeVRo.py --learning_rate 1e-4 --checkpoint_dir ./checkpoints/lstm/l2 --weight_reg l2 --weight_reg_strength 0.18




