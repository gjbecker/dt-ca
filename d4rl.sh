#!/bin/bash
set -e

function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# source $DIR/venv/bin/activate
# export PYTHONPATH=${DIR}/venv/bin/python/dist-packages
# echo "Entered virtualenv."

# Train tf 
print_header "Running D4RL collision avoidance Decision Transformer python script"

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
# cd $DIR
python experiment_d4rl.py