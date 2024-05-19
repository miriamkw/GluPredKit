from sklearn.base import BaseEstimator, TransformerMixin
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
from sklearn.exceptions import NotFittedError


class BaseModel(BaseEstimator, TransformerMixin):
    def __init__(self, prediction_horizon):
        self.prediction_horizon = prediction_horizon
        self.is_fitted = False

    def fit(self, x_train, y_train):
        # Perform any additional processing of the input features here

        # Fit the model
        # ...
        self.is_fitted = True

        raise NotImplementedError("Model has not implemented fit method!")

    def predict(self, x_test):
        # Perform any additional processing of the input features here

        # Raise NotFittedError if predict is called before the model is fitted
        if not self.is_fitted:
            raise NotFittedError("This model instance is not fitted yet. Call 'fit' with appropriate arguments before "
                                 "using this estimator.".format(type(self).__name__))

        # Make predictions using the fitted model
        # ...

        # Return the predictions
        raise NotImplementedError("Model has not implemented predict method!")

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        raise NotImplementedError("Model has not implemented best_params method!")

    def process_data(self, df, model_config_manager: ModelConfigurationManager, real_time: bool):
        # Implement library specific preprocessing steps that are required before training a pandas dataframe
        raise NotImplementedError("Model has not implemented process_data method!")
