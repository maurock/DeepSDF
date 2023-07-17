#!/bin/bash
env_name=$1

echo "$env_name"

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

pip install -e .

# Update the environment
conda env update -n $env_name --file $ENVIRONMENT_FILE

if [[ "$(uname)" == "Linux" ]]; then
    # Install PyTorch
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    # install Pytorch3D
    conda install pytorch3d==0.7.4 -c pytorch3d
fi