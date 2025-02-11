import io

import pandas as pd
import sys
import os
import ast
import requests
import click
import json
import dill
import importlib
from importlib import resources
from pathlib import Path
from ..models.base_model import BaseModel
from ..metrics.base_metric import BaseMetric
from ..helpers.model_config_manager import ModelConfigurationManager


def read_data_from_csv(input_path, file_name):
    input_path = Path(input_path)
    file_path = input_path / file_name
    return pd.read_csv(file_path, index_col="date", parse_dates=True)


def store_data_as_csv(df, output_path, file_name):
    file_path = output_path + file_name
    df.to_csv(file_path)


def split_string(input_string):
    return [] if not input_string else [elem.strip() for elem in input_string.split(',')]


def user_input_prompt(text):
    # Prompt the user for confirmation
    user_response = input(f"{text} (Y/n): ")

    # Convert the user's response to lowercase and check against 'y' and 'n'
    if user_response.lower() == 'y':
        # Continue processing
        print("Continuing processing...")
    elif user_response.lower() == 'n':
        # Stop or exit
        print("Operation aborted by the user.")
        sys.exit()
    else:
        print("Invalid input. Please respond with 'Y' or 'n'.")


def get_metric_module(metric):
    metric_module = importlib.import_module(f'glupredkit.metrics.{metric}')
    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(metric_module.Metric, BaseMetric):
        raise Exception(f"The selected metric '{metric}' must inherit from BaseMetric.")

    return metric_module


def get_model_module(model=None, model_path=None):
    """
    model_module = importlib.import_module(f'glupredkit.models.{model}')
    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(model_module.Model, BaseModel):
        raise Exception(f"The selected model '{model}' must inherit from BaseModel.")
    return model_module
    """
    if model:
        try:
            model_module = importlib.import_module(f'glupredkit.models.{model}')
        except ModuleNotFoundError:
            raise Exception(f"The predefined model '{model}' could not be found.")
        if not issubclass(model_module.Model, BaseModel):
            raise Exception(f"The selected model '{model}' must inherit from BaseModel.")
        return model_module

    elif model_path:
        if not os.path.exists(model_path):
            raise Exception(f"The specified model path '{model_path}' does not exist.")
        model_dir, model_file = os.path.split(model_path)
        module_name, _ = os.path.splitext(model_file)

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        try:
            spec = importlib.util.spec_from_file_location(module_name, model_path)
            custom_model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_model_module)
        except Exception as e:
            raise Exception(f"Failed to load the custom model from '{model_path}': {e}")

        if not hasattr(custom_model_module, 'Model') or not issubclass(custom_model_module.Model, BaseModel):
            raise Exception(f"The custom model in '{model_path}' must define a 'Model' class inheriting from BaseModel.")

        return custom_model_module

    else:
        raise ValueError("Either 'model' or 'model_path' must be provided.")


def get_trained_model(model_file_name):
    model_path = "data/trained_models/"
    with open(model_path + model_file_name, 'rb') as f:
        model_instance = dill.load(f)

    return model_instance


def get_preprocessed_data(prediction_horizon: int, config_manager: ModelConfigurationManager, carbs=None, insulin=None,
                          start_date=None, end_date=None):
    preprocessor = config_manager.get_preprocessor()
    input_file_name = config_manager.get_data()

    print(f"Preprocessing data using {preprocessor} from file data/raw/{input_file_name}, with a prediction "
          f"horizon of {prediction_horizon} minutes...")
    preprocessor_module = importlib.import_module(f'glupredkit.preprocessors.{preprocessor}')
    chosen_preprocessor = preprocessor_module.Preprocessor(config_manager.get_subject_ids(),
                                                           config_manager.get_num_features(),
                                                           config_manager.get_cat_features(),
                                                           config_manager.get_what_if_features(),
                                                           prediction_horizon, config_manager.get_num_lagged_features())
    # Load the input CSV file into a DataFrame
    data = read_data_from_csv("data/raw/", input_file_name)

    # Checking if the data and the configuration are aligned
    required_features = (config_manager.get_num_features() + config_manager.get_cat_features() +
                         config_manager.get_what_if_features())
    columns_list = data.columns.tolist()
    exclude_features = ['id', 'is_test']

    missing_features = [feature for feature in required_features if feature not in columns_list]
    common_elements = [item for item in exclude_features if item in required_features]
    if "CGM" not in config_manager.get_num_features():
        raise ValueError(f"CGM is a required column for numerical features. Please ensure that your configuration and "
                         f"input data are valid.")
    #if missing_features:
    #    raise ValueError(f"The following features are defined in the configuration, but are missing from the data: "
    #                     f"{', '.join(missing_features)}. ")
    #if common_elements:
    #    raise ValueError(f"'id' and 'is_test' should not be used as a features because they are used as columns to "
    #                     f"separate subjects and train and test data. Please remove these features from the "
    #                     f"configuration.")

    if carbs:
        if 'carbs' in data.columns:
            data.at[data.index[-1], 'carbs'] = carbs
        else:
            raise Exception("No input feature named 'carbs'.")

    if insulin:
        if 'insulin' in data.columns:
            data.at[data.index[-1], 'insulin'] = insulin
        else:
            raise Exception("No input feature named 'insulin'.")

    if end_date:
        # Convert the prediction_date string to a datetime object
        end_date = pd.to_datetime(end_date, format="%d-%m-%Y/%H:%M")
        end_date = end_date.tz_localize(data.index.tz)
        nearest_index = abs(data.index - end_date).argmin()
        data = data.iloc[:nearest_index + 1]

    train_data, test_data = chosen_preprocessor(data)

    if start_date:
        # Convert the prediction_date string to a datetime object
        start_date = pd.to_datetime(start_date, format="%d-%m-%Y/%H:%M")
        start_date = start_date.tz_localize(test_data.index.tz)
        nearest_index = abs(test_data.index - start_date).argmin()
        test_data = test_data.iloc[nearest_index:]

    return train_data, test_data


def list_files_in_directory(directory_path):
    file_list = []
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_list.append(filename)
    return file_list


def list_files_in_package(directory):
    package = __import__('glupredkit')
    package_path = Path(resources.files(package) / directory)
    file_paths = [str(file) for file in package_path.iterdir() if file.is_file()]
    file_names = [os.path.basename(path) for path in file_paths]

    return file_names


def validate_config_file_name(ctx, param, file_name):
    file_name = str(file_name)
    base_name = os.path.basename(file_name)
    name_without_extension = os.path.splitext(base_name)[0]
    return name_without_extension


def check_if_data_file_exists(ctx, param, file_name):
    # Function to strip the extension and return the file name without extension
    def strip_extension(file_name):
        return os.path.splitext(os.path.basename(file_name))[0]

    if strip_extension(file_name) == 'synthetic_data':
        return strip_extension(file_name)

    # Check if file exists within a relative path
    if os.path.isfile(file_name):
        return strip_extension(file_name)

    # If it's not a relative path, construct the path using the data/raw/ directory
    data_folder = 'data/raw/'
    full_path = os.path.join(data_folder, file_name)
    if not os.path.isfile(full_path):
        raise ValueError(f"Data file '{file_name}' not found in '{data_folder}' folder. Ensure the file is in the "
                         f"correct directory or provide the full path.")
    return strip_extension(full_path)


def validate_subject_ids(ctx, param, value):
    if value is None or value.strip() == '':
        return []
    try:
        # Convert string representation of a list to an actual list
        value = ast.literal_eval(value)
        # Check if elements are integers, if not, raise ValueError
        if not all(isinstance(val, int) for val in value):
            raise ValueError
        return value
    except (ValueError, SyntaxError):
        raise click.BadParameter("Invalid format for subject ids. List must be a comma-separated list of integers.")


def validate_prediction_horizon(ctx, param, value):
    try:
        value = int(value)
        if value < 10:
            raise ValueError("The prediction horizon must be greater than or equal to 10. Current value: {}".format(value))
    except ValueError:
        raise click.BadParameter('Prediction horizon must be a positive integer.')
    return value


def validate_num_lagged_features(ctx, param, value):
    try:
        value = int(value)
        if value < 0:
            raise ValueError
    except ValueError:
        raise click.BadParameter('The number of time lagged features must be a positive integer.')
    return value


def validate_feature_list(ctx, param, value):
    if value.startswith('[') and value.endswith(']'):
        try:
            # Convert string representation of a list to an actual list
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise click.BadParameter("Invalid format for prediction horizons.")
    try:
        if isinstance(value, str):
            # Return an empty list if the input string is empty after removing spaces
            if value.strip() == '':
                return []
            value = value.replace(' ', '').split(',')
        return value
    except ValueError:
        raise click.BadParameter('List must be a comma-separated list of strings without spaces.')


def validate_test_size(ctx, param, value):
    try:
        test_size = float(value)
        if not 0 <= test_size <= 1:
            raise ValueError
    except ValueError:
        raise click.BadParameter('Test size must be a float between 0 and 1. Decimal values are represented using a '
                                 'period (dot).')
    return test_size


def add_is_test_column(parsed_data, test_size):
    parsed_data['is_test'] = False
    subject_ids = parsed_data['id'].unique()
    processed_dfs = []

    for subject_id in subject_ids:
        subject_data = parsed_data[parsed_data['id'] == subject_id].copy()
        split_index = int(len(subject_data) * (1 - test_size))
        subject_data.iloc[split_index:, subject_data.columns.get_loc('is_test')] = True
        subject_data.sort_index(inplace=True)
        processed_dfs.append(subject_data)

    parsed_data = pd.concat(processed_dfs)
    return parsed_data

