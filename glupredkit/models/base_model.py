from sklearn.base import BaseEstimator, TransformerMixin
from glupredkit.helpers.model_config_manager import ModelConfigurationManager


class BaseModel:
    def __init__(self, prediction_horizon):
        self.prediction_horizon = prediction_horizon

    def fit(self, x_train, y_train):
        # Perform any additional processing of the input features here

        # Fit the model
        # ...

        raise NotImplementedError("Model has not implemented fit method!")

    def predict(self, x_test):
        # Perform any additional processing of the input features here

        # Make predictions using the fitted model
        # ...

        # Return the predictions
        raise NotImplementedError("Model has not implemented predict method!")

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        raise NotImplementedError("Model has not implemented predict method!")

    def process_data(self, df, model_config_manager: ModelConfigurationManager, real_time: bool):
        # Implement library specific preprocessing steps that are required before training a pandas dataframe
        raise NotImplementedError("Model has not implemented predict method!")
