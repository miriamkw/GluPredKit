#!/bin/bash 

# Check if python is installed command 
command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || { echo >&2 "Python required but it's not installed. Aborting."; exit 1; } 

# Set up the virtual environment 
echo "Setting up virtual environment..." 
python -m venv bgp-evaluation 

# Activate the virtual environment 
echo "Activating the virtual environment..." 
source bgp-evaluation/bin/activate 

# Install the required packages 
echo "Installing dependencies..." 
pip install -r requirements.txt 
# Set up submodules 

echo "Initializing and updating submodules..." 
git submodule update --init --recursive 

# Ensure directory structure is in place 
echo "Setting up directory structure..." 

if [ ! -d "data" ]; then 
	mkdir -p data/raw 
	mkdir data/processed 
	mkdir data/trained_models
fi 

if [ ! -d "results" ]; then 
	mkdir -p results/reports 
	mkdir results/figures 
fi 

echo "Installation complete. Now, you're ready to use the Command Line Interface (CLI) for processing and predicting blood glucose levels."
		