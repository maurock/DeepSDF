#!/bin/sh

# Create directories if they do not exist
mkdir -p results/runs

# Add files __init__.py
touch results/runs/__init__.py

# Get objects data
mkdir -p data/
cd data/
wget "https://uob-my.sharepoint.com/:f:/g/personal/ri21540_bristol_ac_uk/EtAq9fvsrDlJmaqWQITgT7cBHYgGW5M-YTq1XHta3xnsBA?e=lmDQfC" -O objects