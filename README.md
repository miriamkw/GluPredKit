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
1. Create a virtual environment by running the following command: `python -m venv bgp-evaluation`.
2. Activate the virtual environment by running the following command: `source bgp-evaluation/bin/activate`.
3. Install the required packages by running the following command: `pip install -r requirements.txt`.

## Usage

### Command line interface

### Running examples

### Implementing BGP Models
To implement your own BGP model, create a new class that inherits from the BaseModel class in `src/models/base_model.py`. 

The `fit` and `predict` methods in the model takes some dataframes as inputs. These dataframes can be custom made or retrieved from real user data using one of the parsers in `src/parsers/`.

The four dataframe inputs represent blood glucose measurements, bolus doses of insulin, basal rate deliveries of insulin and carbohydrate intakes.

In the following tables are the columns and some example data for each one of the dataframes:

Blood glucose measurements (`df_glucose`):

| time                 | units  | value   | device_name |
|----------------------|--------|---------|-------------|
| 2023-02-09T23:57:00Z | mmol/L | 8.43714 | Dexcom G6   |
| 2023-02-09T23:51:59Z | mmol/L | 8.27061 | Dexcom G6   |
| 2023-02-09T23:46:59Z | mmol/L | 8.21511 | Dexcom G6   |

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
| Bayer                                           | Bayer     | Penalty function. The lower, the better.                                                                                                                                                                                                                           | 
| Cao                                             | Cao       | Penalty function. The lower, the better.                                                                                                                                                                                                                           | 
| Kovatchev                                       | Kovatchev | Penalty function. The lower, the better.                                                                                                                                                                                                                           | 
| Van Herpe                                       | VanHerpe  | Penalty function. The lower, the better.                                                                                                                                                                                                                           | 
| Temporal Gain (to do!)                          | TG        | Takes prediction horizon into account. The amount of average time gained for early detection of a potential hypo/hyper glycemia event using the model (https://ieeexplore.ieee.org/document/6157604). The higher value, the better.                                | 
| Energy of the second-order differences (to do!) | ESOD      | Takes prediction horizon into account. Reflects the presence of spurious oscillations in the predicted time series, and thus the risk of generating false hypo/hyper alerts. The closer to 1, the better (https://ieeexplore.ieee.org/document/6157604).           | 
| J (to do!)                                      | J         | Takes prediction horizon into account. Simultaneously takes into account two merit criteria (TG and ESOD), the regularity of the predicted profile and the time gained thanks to prediction (https://ieeexplore.ieee.org/document/6157604). The lower, the better. | 
| Pearson's Correlation Coefficient               | PCC       | a measure of the linear relationship between two variables X and Y, giving a value between -1 and +1. A value of +1 indicates a perfect positive correlation, 0 indicates no correlation, and -1 indicates a perfect negative correlation.                         | 

## Disclaimers
* Datetimes that are fetched from Tidepool API are received converted to timezone offset +00:00. There is no way to get information about the original timezone offset from this data source.
* Bolus doses that are fetched from Tidepool API does not include the end date of the dose delivery.
* Metrics assumes mg/dL for the input.
