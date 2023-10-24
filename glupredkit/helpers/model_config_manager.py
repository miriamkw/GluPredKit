"""
This file defines the schema and provides get methods for the configuration files describing the properties of the
models before preprocessing and training.
"""

import json


class ModelConfigurationManager:
    def __init__(self, config_file):
        self.config_file = 'data/configurations/' + config_file + '.json'
        self.schema = {
            "data": str,
            "preprocessor": str,
            "prediction_horizons": list,
            "num_lagged_features": int,
            "num_features": list,
            "cat_features": list,
            "test_size": float
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

    def get_preprocessor(self):
        return self.config["preprocessor"]

    def get_prediction_horizons(self):
        return self.config["prediction_horizons"]

    def get_num_lagged_features(self):
        return self.config["num_lagged_features"]

    def get_num_features(self):
        return self.config["num_features"]

    def get_cat_features(self):
        return self.config["cat_features"]

    def get_test_size(self):
        return self.config["test_size"]

