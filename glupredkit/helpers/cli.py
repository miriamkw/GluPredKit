import pandas as pd
import sys
import os
import dill
import importlib
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
    return [] if not input_string else input_string.split(',')


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
    chosen_preprocessor = preprocessor_module.Preprocessor(config_manager.get_num_features(),
                                                           config_manager.get_cat_features(),
                                                           prediction_horizon, config_manager.get_num_lagged_features(),
                                                           config_manager.get_test_size())
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
