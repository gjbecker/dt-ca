#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/decision_transformer/envs/gym_ca_data_gen/gym_collision_avoidance/experiments/utils.sh

# Train tf 
print_header "Running grid Decision Transformer python script"

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
# cd $DIR
python experiment_grid.py