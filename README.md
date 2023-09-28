# Blood Glucose Prediction Evaluation

This repository provides a framework for training, testing, and evaluating blood glucose prediction (BGP) models. The following features are provided:
* Fetch user data from Tidepool (more data sources might be implemented in future versions).
* Examples of BGP models in `loop_model_scoring/models`.
* Base class for BGP models where users can implement their own prediction models.
* Examples of BGP evaluation metrics in `loop_model_scoring/metrics`.
* Base class for BGP evaluation metrics where users can implement their own evaluation metrics.
* Graphic visualization alternatives of the performance of BGP.

## Content
* [Setup](#setup)
* [Usage](#usage)
  * [Command line interface](#command-line-interface)
  * [Running examples](#running-examples)
  * [Implementing BGP models](#implementing-bgp-models)
  * [Implementing BGP evaluation metrics](#implementing-bgp-evaluation-metrics)
  * [Testing](#testing)
* [Error Metrics Overview](#error-metrics-overview)
* [Disclaimers](#disclaimers)

## Setup
1. Create a virtual environment by running the following command: `python -m venv bgp-evaluation` or `python3 -m venv bgp-evaluation`.
2. Activate the virtual environment by running the following command: `source bgp-evaluation/bin/activate`.
3. Install the required packages by running the following command: `pip install -r requirements.txt`.

## Usage

### Command line interface

### Running examples

#### Credentials
Create a file named `credentials.json` in the root directory. Copy and paste the following information and adjust the information with your credentials:

`{ "tidepool_api": 
    { "email": "YOUR_TIDEPOOL_USERNAME", 
    "password": "YOUR_TIDEPOOL_PASSWORD" 
  },
  "nightscout_api": {
    "url": "https://diabetes.neethan.net/",
    "api_secret": "wD6KB2HvJ5ZL3FZphKjbPLdHb5C1zEix"
  }
}`



### Implementing BGP Models
To implement your own BGP model, create a new class that inherits from the BaseModel class in `src/models/base_model.py`. 

The `fit` and `predict` methods in the model takes some dataframes as inputs. These dataframes can be custom made or retrieved from real user data using one of the parsers in `src/parsers/`.

The four dataframe inputs represent blood glucose measurements, bolus doses of insulin, basal rate deliveries of insulin and carbohydrate intakes.

In the following tables are the columns and some example data for each one of the dataframes:

Blood glucose measurements (`df_glucose`):

| time                 | units  | value | device_name |
|----------------------|--------|-------|-------------|
| 2023-02-09T23:57:00Z | mg/dL  | 105.3 | Dexcom G6   |
| 2023-02-09T23:51:59Z | mg/dL  | 106.1 | Dexcom G6   |
| 2023-02-09T23:46:59Z | mg/dL  | 106.9 | Dexcom G6   |

Note that all parsers default return glucose values in mg/dL.

Bolus doses of insulin (`df_bolus`):

| time                 | dose[IU] | device_name  | 
|----------------------|----------|--------------|
| 2023-02-09T23:57:00Z | 1        | Omnipod-Dash |
| 2023-02-09T23:51:59Z | 2.5      | Omnipod-Dash |
| 2023-02-09T23:46:59Z | 2.1      | Omnipod-Dash |


Basal rates of insulin (`df_basal`):

| time                 | duration[ms] | rate[U/hr] | device_name   | scheduled_basal | delivery_type |
|----------------------|--------------|------------|---------------|-----------------|---------------|
| 2023-02-09T23:57:00Z | 4505492      | 0.759073   | Omnipod-Dash  | 0.75 IU/hr      | basal         |
| 2023-02-09T23:51:59Z | 133717       | 0.0        | Omnipod-Dash  | 0.75 IU/hr      | temp          | 
| 2023-02-09T23:46:59Z | 1198603      | 0.0        | Omnipod-Dash  | 0.75 IU/hr      | temp          |

Carbohydrate intakes (`df_carbs`):

| time                 | units  | value | absorption_time\[s] |
|----------------------|--------|-------|---------------------|
| 2023-02-09T23:57:00Z | grams  | 30    | 10800               |
| 2023-02-09T23:51:59Z | grams  | 50    | 10800               |
| 2023-02-09T23:46:59Z | grams  | 45    | 10800               |



### Implementing BGP Evaluation Metrics
To implement your own BGP evaluation metric, create a new class that inherits from the BaseMetric class in `src/metrics/base_metric.py`. Your new class should implement the `__call__` method, which takes two lists of glucose values (the true values and the predicted values) as input and returns a single value representing the performance of the metric.

### Testing
To run the tests, write `python tests/test_all.py` in the terminal.

## Error Metrics Overview

| Name                                            | Class     | Description                                                                                                                                                                                                                                                        |
|-------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Root Mean Squared Error                         | RMSE      | Returns a value between [0, inf]. Treats high and low values equally.                                                                                                                                                                                              | 
| Mean Absolute Error                             | MAE       | Returns a value between [0, inf]. Treats high and low values equally.                                                                                                                                                                                              | 
| Pearson's Correlation Coefficient               | PCC       | a measure of the linear relationship between two variables X and Y, giving a value between -1 and +1. A value of +1 indicates a perfect positive correlation, 0 indicates no correlation, and -1 indicates a perfect negative correlation.                         | 

## Disclaimers
* Datetimes that are fetched from Tidepool API are received converted to timezone offset +00:00. There is no way to get information about the original timezone offset from this data source.
* Bolus doses that are fetched from Tidepool API does not include the end date of the dose delivery.
* Metrics assumes mg/dL for the input.
