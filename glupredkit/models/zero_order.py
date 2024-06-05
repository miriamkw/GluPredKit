"""
A zero-order hold algorithm assumes that glucose will not change in the future.
"""
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
import numpy as np


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

    def _fit_model(self, x_train, y_train, *args):
        return self

    def _predict_model(self, x_test):
        n_predictions = self.prediction_horizon // 5
        cgm_values = x_test['CGM'].tolist()

        # Replicate each value in the "CGM" column n_predictions times
        y_pred = [[cgm_value] * n_predictions for cgm_value in cgm_values]
        return np.array(y_pred)

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)
