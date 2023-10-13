#!/bin/bash 
# Automatically exit script if an error appears
set -e

# Check if python is installed command
if ! command -v python3 &> /dev/null
then
    echo "python3 required but could not be found. Aborting."
    exit
fi

# Set up the virtual environment 
echo "Setting up virtual environment..." 
python3 -m venv bgp-evaluation

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
	mkdir data/.processed
	mkdir data/.trained_models
	mkdir data/figures
	mkdir data/reports
fi

echo "Installation complete. Now, you're ready to use the Command Line Interface (CLI) for processing and predicting blood glucose levels."
		