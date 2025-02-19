from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
import numpy as np


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.is_fitted = True

    def _fit_model(self, x_train, y_train, *args):
        return self

    def _predict_model(self, x_test):

        """
        Apply prediction function to each row of a DataFrame.

        Args:
            x_test (pd.DataFrame): DataFrame where each row contains exactly three numerical values.

        Returns:
            list of lists: Lists containing predicted trajectories for input each row.
        """
        steps = self.prediction_horizon // 5
        predictions = x_test.apply(lambda row: self.predict_future_row(row, steps), axis=1)

        return predictions

    def predict_future_row(self, row, steps=1):
        """
        Predict future values based on the slope of the last three values in a row.

        Args:
            row (pd.Series): A row of the DataFrame with exactly three numerical values.
            steps (int): Number of future steps to predict.

        Returns:
            list: Predicted future values.
        """
        # Compute slopes
        diff1 = row['CGM_10'] - row['CGM_15']
        diff2 = row['CGM_5'] - row['CGM_10']
        diff3 = row['CGM'] - row['CGM_5']

        slope = np.mean([diff1, diff2, diff3])

        # Predict future values
        predictions = [row['CGM']]
        for _ in range(steps):
            next_value = predictions[-1] + slope
            predictions.append(next_value)

        return predictions[1:]  # Remove the last known value


    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def print_coefficients(self):
        feature_names = self.model.feature_names_in_
        coefficients = self.model.coef_
        for feature_name, coefficient in zip(feature_names, coefficients):
            print(f"Feature: {feature_name}, Coefficient: {coefficient:.4f}")

