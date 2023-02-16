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
Run `<EXAMPLE_SCRIPT>.py` in terminal in the root directory to recreate the loop scoring plot examples used in the documentation.

#### example_mock_data.py
An example script that recreates the examples in the documentation of the penalty function.

#### example_one_forecast.py
An example script that calculates and plots one forecast trajectory for a given date, and calculates the penalty for the trajectory.

#### example_between_dates.py
An example script that calculates and plots the penalties for the forecasts between two given dates.


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


# Important Notes
- Predictions are based solemnly on past data (assuming no knowledge about future inputs)
- For now, the default meal model is the linear one in pyloopkit, and there is no simple way of using parabolic (I plan to change it and add a PR)


