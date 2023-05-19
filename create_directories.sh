#!/bin/sh

# Create directories if they do not exist
mkdir -p results/runs_sdf
mkdir -p results/runs_touch
mkdir -p results/runs_touch_sdf
mkdir -p data/objects
mkdir -p data/ShapeNetCoreV2urdf


# Add files __init__.py
touch results/runs_sdf/__init__.py
touch results/runs_touch/__init__.py
touch results/runs_touch_sdf/__init__.py
touch data/objects/__init__.py
touch data/ShapeNetCoreV2urdf/__init__.py
touch examples/__init__.py