# Blood Glucose Prediction Evaluation

This repository provides a framework for training, testing, and evaluating blood glucose prediction (BGP) models. The following features are provided:
* Fetch user data from Tidepool (more data sources might be implemented in future versions).
* Examples of BGP models in `loop_model_scoring/models`.
* Base class for BGP models where users can implement their own prediction models.
* Examples of BGP evaluation metrics in `loop_model_scoring/metrics`.
* Base class for BGP evaluation metrics where users can implement their own evaluation metrics.
* Graphic visualization alternatives of the performance of BGP.

## Content (to do: create smart links)
* Prerequisites
* Setup
* Usage
  * Command line interface
  * Running examples
  * Implementing BGP models
  * Implementing BGP evaluation metrics
  * Testing
* [Error Metrics Overview](#error-metrics-overview)
* Disclaimers

## Setup
1. Create a virtual environment by running the following command: python -m venv bgp-evaluation 
2. Activate the virtual environment by running the following command: source bgp-evaluation/bin/activate 
3. Install the required packages by running the following command: pip install -r requirements.txt

## Usage

### Implementing BGP Models

### Implementing BGP Evaluation Metrics
To implement your own BGP evaluation metric, create a new class that inherits from the BaseMetric class in `loop_model_scoring/metrics/base_metric.py`. Your new class should implement the evaluate method, which takes two lists of glucose values (the true values and the predicted values) as input and returns a single value representing the performance of the metric.

### Testing
To run the tests, write `python tests/test_all.py` in the terminal.

## Error Metrics Overview

| Name                    | Class | Description | Weaknesses |
|-------------------------|-------|-------------|------------|
| Root Mean Squared Error | RMSE  |             |            | 
| Mean Absolute Error     | MAE   |             |            |
| Error                   | 45    |             |            |






