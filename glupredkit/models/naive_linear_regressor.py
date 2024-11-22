from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from sklearn.multioutput import MultiOutputRegressor
import numpy as np


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.columns = ['CGM_15', 'CGM_10', 'CGM_5', 'CGM']
        self.model = None

    def _fit_model(self, x_train, y_train, *args):
        return self

    def _predict_model(self, x_test):
        interval = 5  # Interval between predictions in minutes
        num_predictions = self.prediction_horizon // interval
        y_pred = []

        for index, row in x_test.iterrows():
            # Extract data for the last 15 minutes (e.g., assuming rows are time-ordered)
            recent_data = row[self.columns][::-1]  # Replace with appropriate logic for 15 min extraction
            times = np.arange(len(recent_data))  # Use relative time indices (e.g., 0, 1, ..., n-1)

            # Reshape data for regression model
            X = times.reshape(-1, 1)
            y = recent_data.values

            # Fit a linear regression model
            reg = LinearRegression()
            reg.fit(X, y)

            # Predict the value at the prediction horizon
            future_times = np.arange(1, num_predictions + 1)
            predictions = reg.predict(future_times.reshape(-1, 1))

            predictions = [0 if val < 0 else 600 if val > 600 else val for val in predictions]

            # Append prediction
            y_pred.append(predictions)

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

