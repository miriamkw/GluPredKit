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


