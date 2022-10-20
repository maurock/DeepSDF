#!/bin/sh

# Create directories if they do not exist
mkdir -p results/runs

# Add files __init__.py
touch results/runs/__init__.py

# Get objects data
mkdir -p data/
cd data/
wget "https://uob-my.sharepoint.com/:u:/g/personal/ri21540_bristol_ac_uk/EWC1KJPvUIVOv14XcnhB7NsBxwq9lmeCe9uVHbGWK0zGZA?e=kfbiRm" -O objects.zip
unzip objects.zip -d data/