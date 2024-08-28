from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from sklearn.multioutput import MultiOutputRegressor
import numpy as np


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.columns = ['CGM', 'CGM_5', 'CGM_10']
        self.model = None

    def _fit_model(self, x_train, y_train, *args):
        # Define the base regressor
        base_regressor = LinearRegression()

        # Wrap the base regressor with MultiOutputRegressor
        multi_output_regressor = MultiOutputRegressor(base_regressor)

        # Perform grid search to find the best parameters and fit the model
        multi_output_regressor.fit(x_train[self.columns], y_train)

        self.model = multi_output_regressor
        return self

    def _predict_model(self, x_test):
        y_pred = self.model.predict(x_test[self.columns])
        y_pred = np.array(y_pred)
        return y_pred

    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def print_coefficients(self):
        feature_names = self.model.feature_names_in_
        coefficients = self.model.coef_
        for feature_name, coefficient in zip(feature_names, coefficients):
            print(f"Feature: {feature_name}, Coefficient: {coefficient:.4f}")

