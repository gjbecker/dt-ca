#!/bin/bash
set -e

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GYM_DIR="$DIR/decision_transformer/envs/gym_ca"

# Install this pkg and its requirements
python -m pip install -e $GYM_DIR

# Install RVO and its requirements
cd $GYM_DIR/gym_collision_avoidance/envs/policies/Python-RVO2
python -m pip install Cython
if [[ "$OSTYPE" == "darwin"* ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.15
    brew install cmake || true
fi
python setup.py build
python setup.py install

echo "Finished installing gym_collision_avoidance!"

echo "Finished installing dt_ca!"
