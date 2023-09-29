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

## Usage of Command Line Interface (CLI)

A Command Line Interface (CLI) is developed to facilitate the process of fetching data, preprocessing data, training prediction models to report the results. Commands are executed from the `src/` directory.

Definitions:
- **Parsing**: Refers to the fetching of data from data sources (for example Nighscout, Tidepool or Apple Health), and to process the data into the same table. The parsed datasets are stored in 'data/raw/'.
- **Preprocessing**: Refers to the preprocessing of the raw datasets from the parsing-stage. This includes imputation, feature addition, removing NaN values, splitting data etc. The preprocessed datasets are stored in 'data/preprocessed'.
- **Model training**: Refers to using preprocessed data to train a blood glucose prediction model. The trained models are stored in 'data/models/'.
- **Metrics**: Refers to different 'scores' to describing the accuracy of the predictions of a blood glucose prediction model. The evaluation metrics are stored in tables 'results/reports/'.
- **Plots**: Different types of plots that can illustrate blood glucose predictions together with actual measured values. The plotted results are stored in 'results/figures/'.

### Getting Started
Make sure you are located in the `src/` directory in the terminal, where `cli.py` is located. 

### Parse Command

The `parse` command is used to parse data using a selected parser and store it as a CSV file in the "data/raw/" directory. 

`python cli.py parse --parser <parser> <username> <password> [--file-name <file-name>]`

#### Options
`--parser`: Choose a parser for data parsing. Supported parsers include 'tidepool' and 'nightscout'.

`--file-name`: (Optional) Specify an optional file name for the output CSV file.

#### Arguments
`<username>`: The username required for data parsing.

`<password>`: The password required for data parsing.

#### Example

To parse data using the 'tidepool' parser with a custom output file name:

`python cli.py parse --parser tidepool my_username my_password --file-name custom_output.csv`

#### Example Output

The parsed data will be stored as a .csv and look something like the table below:

| date                      | CGM        | insulin | carbs |
|---------------------------|------------|---------|-------|
| 2023-03-01 02:30:00+02:00 | 150.021695 | 0.06    | 30    |    
| 2023-03-01 02:35:00+02:00 | 146.021114 | 0.06    | 0     |      
| 2023-03-01 02:40:00+02:00 | 143.020724 | 0.06    | 0     |

Additional columns is possible. 

## Contributing with code

TODO: Describe the file structure.

### Adding Data Source Parsers

### Adding Data Preprocessors

### Adding Machine Learning Prediction Models

### Adding Evaluation Metrics
To implement your own BGP evaluation metric, create a new class that inherits from the BaseMetric class in `src/metrics/base_metric.py`. Your new class should implement the `__call__` method, which takes two lists of glucose values (the true values and the predicted values) as input and returns a single value representing the performance of the metric.


### Adding Evaluation Plots



### Testing
To run the tests, write `python tests/test_all.py` in the terminal.

## Error Metrics Overview

| Name                                            | Class     | Description                                                                                                                                                                                                                                                        |
|-------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Root Mean Squared Error                         | RMSE      | Returns a value between [0, inf]. Treats high and low values equally.                                                                                                                                                                                              | 
| Mean Absolute Error                             | MAE       | Returns a value between [0, inf]. Treats high and low values equally.                                                                                                                                                                                              | 
| Pearson's Correlation Coefficient               | PCC       | a measure of the linear relationship between two variables X and Y, giving a value between -1 and +1. A value of +1 indicates a perfect positive correlation, 0 indicates no correlation, and -1 indicates a perfect negative correlation.                         | 

## Disclaimers and limitations
* Datetimes that are fetched from Tidepool API are received converted to timezone offset +00:00. There is no way to get information about the original timezone offset from this data source.
* Bolus doses that are fetched from Tidepool API does not include the end date of the dose delivery.
* Metrics assumes mg/dL for the input.
* Note that the difference between how basal rates are registered. Bolus doses are however consistent across. Hopefully it is negligable.