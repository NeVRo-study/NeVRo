#!/bin/bash

### Task: LSTM-Network
## Training
python3 NeVRo.py --max_steps 15 --checkpoint_dir ./LSTM/checkpoints/lstm/l2 --weight_reg l2 --weight_reg_strength 0.18

# python2 NeVRo.py --learning_rate 1e-4 --checkpoint_dir ./checkpoints/lstm/l1 --weight_reg l1 --weight_reg_strength 0.18
# python2 NeVRo.py --learning_rate 1e-4 --checkpoint_dir ./checkpoints/lstm/l2 --weight_reg l2 --weight_reg_strength 0.18




