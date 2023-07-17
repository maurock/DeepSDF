#!/bin/bash

# Check the operating system
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    ENVIRONMENT_FILE="environment_mac.yaml"
elif [[ "$(uname)" == "Linux" ]]; then
    # Linux
    ENVIRONMENT_FILE="environment_linux.yaml"
else
    echo "Unsupported operating system. Please install the dependencies manually."
    exit 1
fi

# Activate Conda environment (replace 'your_environment_name' with your actual environment name)
conda activate deepsdf

pip install .

# Update the environment
conda env update -n deepsdf --file $ENVIRONMENT_FILE

if [[ "$(uname)" == "Linux" ]]; then
    # Install PyTorch
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    # install Pytorch3D
    conda install pytorch3d==0.7.4 -c pytorch3d