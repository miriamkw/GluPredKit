"""
This file defines the schema and provides get methods for the configuration files describing the properties of the
models before preprocessing and training.
"""

import json
import os


def generate_model_configuration(file_name, data, subject_ids, preprocessor, prediction_horizon, num_lagged_features,
                                 num_features, cat_features, what_if_features):
    # Check if the 'data' string is a valid file in the 'data/raw/' folder
    data_path = os.path.join('data/raw', data + '.csv')
    if not os.path.isfile(data_path):
        raise ValueError(f"Data file '{data + '.csv'}' not found in 'data/raw/' folder.")

    # Check if 'preprocessor' is a valid preprocessor module
    preprocessor_module = f'glupredkit.preprocessors.{preprocessor}'
    try:
        __import__(preprocessor_module)
    except ImportError:
        raise ValueError(f"Preprocessor '{preprocessor}' not found in 'preprocessors' module.")

    config = {
        "data": data + '.csv',
        "subject_ids": subject_ids,
        "preprocessor": preprocessor,
        "prediction_horizon": prediction_horizon,
        "num_lagged_features": num_lagged_features,
        "num_features": num_features,
        "cat_features": cat_features,
        "what_if_features": what_if_features
    }
    # Save the generated config to a JSON file
    with open(f'data/configurations/{file_name}.json', 'w') as f:
        json.dump(config, f, indent=4)


class ModelConfigurationManager:
    def __init__(self, config_file):
        self.config_file = 'data/configurations/' + config_file + '.json'
        self.schema = {
            "data": str,
            "subject_ids": list,
            "preprocessor": str,
            "prediction_horizon": int,
            "num_lagged_features": int,
            "num_features": list,
            "cat_features": list,
            "what_if_features": list
        }
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.validate_config(config)
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{self.config_file}' not found.")

    def validate_config(self, config):
        for key, value_type in self.schema.items():
            if key not in config:
                raise ValueError(f"Missing key '{key}' in the config file.")
            if not isinstance(config[key], value_type):
                raise ValueError(f"Invalid value type for '{key}' in the config file. Expected {value_type}.")

    def get_data(self):
        return self.config["data"]

    def get_subject_ids(self):
        return self.config["subject_ids"]

    def get_preprocessor(self):
        return self.config["preprocessor"]

    def get_prediction_horizon(self):
        return self.config["prediction_horizon"]

    def get_num_lagged_features(self):
        return self.config["num_lagged_features"]

    def get_num_features(self):
        return self.config["num_features"]

    def get_cat_features(self):
        return self.config["cat_features"]

    def get_what_if_features(self):
        return self.config["what_if_features"]
