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

#### Description

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

### Preprocess Command

#### Description

The `preprocess` command allows you to preprocess data from an input CSV file (must be parsed, and located in 'data/raw/') and store train and test data into CSV files. You can choose the preprocessor and specify various options for preprocessing.

`python cli.py preprocess [OPTIONS] INPUT-FILE-NAME`

#### Options

`--preprocessor`: Choose the preprocessor type (default: scikit_learn). Available options are dynamically generated based on the parsers found in the parsers folder.
`INPUT-FILE-NAME`: Input CSV file containing the data.

#### Additional Options

`--prediction-horizon`: The prediction horizon for the target value in minutes (default: 60). Must be divisible by 5.
`--num-lagged-features`: The number of samples of time-lagged features (default: 12).
`--include-hour`: Include the hour of the day as an input feature (default: True).
`--test-size`: Fraction of data to reserve for testing (default: 0.2).

#### Example

`python cli.py preprocess --preprocessor scikit_learn my_data.csv`

This command will preprocess a file named 'my_data.csv' in 'data/raw/' and save the training and testing datasets in the 'data/processed/' directory with filenames indicating the preprocessor and selected options. The generated files would be stored as:

- `train-data_scikit_learn_ph-60_lag-12.csv`
- `test-data_scikit_learn_ph-60_lag-12.csv`

#### Notes
- Ensure that the 'data/raw/' folder contains the necessary file for the given file name.
- The prediction horizon must be divisible by 5.

### Train Model Command

#### Description
This command trains the model with the given options and arguments, saves the trained model instance, and prints out the trained model's hyperparameters if available.

#### Usage
```sh
python cli.py --model [MODEL] INPUT-FILE-NAME --prediction-horizon [PREDICTION_HORIZON] --num_features [NUM_FEATURES] --cat_features [CAT_FEATURES]
```

#### Options and Arguments
- `--model`: The name of the model file to be trained (without `.py`).
    - This is a required option.
    - Example: `--model huber`

- `INPUT-FILE-NAME` (Positional Argument): The name of the input file (located in `../data/processed/`) containing the training data.
    - This is a required argument.
    - Example: `input_file.csv`

- `--prediction-horizon`: The prediction horizon for the model.
    - Default is `60`.
    - Example: `--prediction-horizon 30`

- `--num_features`: List of numerical features, separated by a comma.
    - Default is `'CGM,insulin,carbs'`.
    - Example: `--num_features feature1,feature2`

- `--cat_features`: List of categorical features, separated by a comma.
    - Default is an empty string `''` (i.e., no categorical features).
    - Example: `--cat_features category1,category2`

#### Example Command
```sh
python cli.py --model huber input_file.csv --prediction-horizon 30 --num_features CGM,insulin --cat_features carbs
```

#### Outputs
- Prints out the progress and status of model training.
- If available, prints out the model hyperparameters.
- Saves the trained model instance to `../data/models/` directory with the name in the format `[model]_ph-[prediction-horizon].pkl`.

#### Notes
- Ensure the input file is correctly placed in the `../data/processed/` directory.
- The selected model must inherit from `BaseModel`.


## Contributing with code

TODO: Describe the file structure.

### Adding Data Source Parsers
Note: Parser must have class name Parser.

Note: add new file to the CLI alternatives.

### Adding Data Preprocessors
Note: targets from preprocessors will be named "target".

Note: Preprocessors must have class name Preprocessor

Note: add new file to the CLI alternatives.

### Adding Machine Learning Prediction Models
Note: targets from presossors will be named "target".

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