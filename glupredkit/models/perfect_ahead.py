from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from sklearn.multioutput import MultiOutputRegressor
import numpy as np


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

    def _fit_model(self, x_train, y_train, *args):
        return self

    def _predict_model(self, x_test):
        # TODO: Get a way of getting access to the target!
        return x_test

    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)


