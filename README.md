# Loop Model Scoring
This repository contains an implementation of a method for scoring the loop model and treatment decisions, as proposed by Damon Bayer in [this documentation](https://docs.google.com/document/d/14AJ9u2oGJiiJU1cWVDf_rC_WdJc0oOj1uIkXutOovQU/edit#).


## Prerequisites

- Python 3.x
- Virtual environment tool


## Setup

1. Create a virtual environment by running the following command:
`python -m venv loop-scoring`

2. Activate the virtual environment by running the following command:
`source loop-scoring/bin/activate`

3. Install the required packages by running the following command:
`pip install -r requirements.txt`


## Usage

The documentation is under construction.

### Run example script
Run `python3 example.py` in terminal in the root directory to recreate the loop scoring plot examples used in the documentation.

### Credentials for Tidepool API

Create a file named `credentials.json` in the root directory. Copy and paste the following information and adjust the information with your credentials:

`{
	"tidepool_api": {
		"email": "YOUR_TIDEPOOL_USERNAME",
		"password": "YOUR_TIDEPOOL_PASSWORD"
	} 
}`

### Therapy settings

The calculations of forecasts are based on the therapy settins defined in `therapy_settings.json`.

