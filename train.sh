#!/bin/bash

# Activate the rl_loco conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_loco

# Set library paths for IsaacGym CUDA support (WSL)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Change to legged_gym directory
cd ~/legged_gym

# Run training with provided arguments or defaults
python legged_gym/scripts/train.py "$@"
