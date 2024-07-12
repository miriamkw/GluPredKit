import numpy as np
import pandas as pd
import sys
import ctypes
import os
import json
import ast
import click
import dill
import importlib
from importlib import resources
from pathlib import Path
from ..models.base_model import BaseModel
from ..metrics.base_metric import BaseMetric
from ..helpers.model_config_manager import ModelConfigurationManager


def read_data_from_csv(input_path, file_name):
    file_path = input_path + file_name
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


def get_model_module(model):
    model_module = importlib.import_module(f'glupredkit.models.{model}')
    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(model_module.Model, BaseModel):
        raise Exception(f"The selected model '{model}' must inherit from BaseModel.")

    return model_module


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
    file_names = [path.split('/')[-1] for path in file_paths]
    return file_names


def validate_file_name(ctx, param, value):
    file_name = str(value)
    # Removing the file extension, if any
    return file_name.partition('.')[0]


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
        if value < 0:
            raise ValueError
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


def add_derived_features(parsed_data, derived_features):
    if 'hour' in derived_features:
        parsed_data['hour'] = parsed_data.index.tz_localize(None).hour
    if 'active_insulin' or 'active_carbs' in derived_features:
        parsed_data['active_insulin'] = np.nan
        parsed_data['active_carbs'] = np.nan

        for i in range(72, len(parsed_data)):
            subset_df = parsed_data.iloc[i-72:i]
            active_insulin, active_carbs = get_active_insulin_and_carbs(subset_df)

            parsed_data.at[parsed_data.index[i], 'active_insulin'] = active_insulin
            parsed_data.at[parsed_data.index[i], 'active_carbs'] = active_carbs

    return parsed_data


def get_active_insulin_and_carbs(subset_df):
    json_data = get_json_from_df(subset_df)
    json_bytes = json_data.encode('utf-8')

    # Load the shared library
    swift_lib = ctypes.CDLL('./libLoopAlgorithmToPython.dylib')

    # Set input and return types
    swift_lib.getActiveInsulin.argtypes = [ctypes.c_char_p]
    swift_lib.getActiveInsulin.restype = ctypes.c_double
    swift_lib.getActiveCarbs.argtypes = [ctypes.c_char_p]
    swift_lib.getActiveCarbs.restype = ctypes.c_double

    # Call the Swift functions
    result_active_insulin = swift_lib.getActiveInsulin(json_bytes)
    result_active_carbs = swift_lib.getActiveCarbs(json_bytes)

    return result_active_insulin, result_active_carbs


def get_json_from_df(subset_df):
    subset_df = subset_df.reset_index()

    df_glucose = subset_df[['date', 'CGM']].copy()
    df_glucose = df_glucose.dropna(subset=['CGM'])  # Drop rows where 'CGM' is NaN
    df_glucose['CGM'] = df_glucose['CGM'].astype(int)
    df_glucose['date'] = df_glucose['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df_glucose.rename(columns={"CGM": "value"}, inplace=True)
    df_glucose.sort_values(by='date', inplace=True, ascending=True)

    # Dataframe carbohydrates
    df_carbs = subset_df[['date', 'carbs']].copy()
    df_carbs = df_carbs[df_carbs['carbs'] > 0]
    df_carbs['absorptionTime'] = 10800
    df_carbs.rename(columns={"carbs": "grams"}, inplace=True)
    df_carbs.sort_values(by='date', inplace=True, ascending=True)
    df_carbs['date'] = df_carbs['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Dataframe bolus doses
    df_bolus = subset_df[['date', 'bolus']].copy()
    df_bolus = df_bolus[df_bolus['bolus'] > 0]
    df_bolus.rename(columns={"date": "startDate", "bolus": "volume"}, inplace=True)
    df_bolus.sort_values(by='startDate', inplace=True, ascending=True)
    df_bolus['startDate'] = df_bolus['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df_bolus['endDate'] = df_bolus['startDate']
    df_bolus['type'] = "bolus"

    # Dataframe basal rates
    df_basal = subset_df[['date', 'basal']].copy()
    df_basal.rename(columns={"date": "startDate", "basal": "volume"}, inplace=True)
    df_basal.sort_values(by='startDate', inplace=True, ascending=True)
    df_basal['endDate'] = df_basal['startDate'] + pd.Timedelta(minutes=5)
    df_basal['startDate'] = df_basal['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df_basal['endDate'] = df_basal['endDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df_basal['type'] = "basal"

    df_insulin = pd.concat([df_bolus, df_basal], ignore_index=True)
    df_insulin.sort_values(by='startDate', inplace=True, ascending=True)

    insulin_history_list = df_insulin.to_dict(orient='records')
    glucose_history_list = df_glucose.to_dict(orient='records')
    carbs_history_list = df_carbs.to_dict(orient='records')

    basal = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": 0.0}]
    carb_ratio = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": 10}]
    ins_sens = [{"endDate": "2030-06-23T05:00:00Z", "startDate": "2020-06-22T10:00:00Z", "value": 60}]

    # Create the final JSON structure
    data = {
        "carbEntries": carbs_history_list,
        "doses": insulin_history_list,
        "glucoseHistory": glucose_history_list,
        "basal": basal,
        "carbRatio": carb_ratio,
        "sensitivity": ins_sens
    }
    data = {**data, **get_recommendation_settings(df_glucose.iloc[-1].loc['date'])}

    # Convert the data dictionary to a JSON string
    return json.dumps(data, indent=2)


def get_recommendation_settings(prediction_start):
    data = {
        "predictionStart": prediction_start,
        "maxBasalRate": 4.1,
        "maxBolus": 9,
        "target": [
            {
                "endDate": "2025-10-18T03:10:00Z",
                "lowerBound": 101,
                "startDate": "2022-10-17T20:59:03Z",
                "upperBound": 115
            }
        ],
        # Are these necessary?
        "recommendationInsulinType": "novolog",
        "recommendationType": "automaticBolus",
    }
    return data

