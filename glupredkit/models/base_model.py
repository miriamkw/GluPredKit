from sklearn.base import BaseEstimator, TransformerMixin
from glupredkit.helpers.model_config_manager import ModelConfigurationManager
from sklearn.exceptions import NotFittedError
from abc import ABC, abstractmethod


class BaseModel(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, prediction_horizon):
        self.prediction_horizon = prediction_horizon
        self.is_fitted = False
        self.model = None  # Placeholder for the fitted model

    def fit(self, x_train, y_train, *args, **kwargs):
        """
        Fits the model to the training data.

        Args:
            x_train (DataFrame): Processed data for model training.
            y_train (DataFrame): Processed target data, where each feature is named target_5, target_10... The number
                behind target corresponds to the prediction horizon
            *args: Additional positional arguments for the fit method.
            **kwargs: Additional keyword arguments for the fit method.

        Returns:
            self: The fitted model instance.
        """
        self.is_fitted = True
        return self._fit_model(x_train, y_train, *args, **kwargs)

    @abstractmethod
    def _fit_model(self, x_train, y_train, *args, **kwargs):
        raise NotImplementedError("Model has not implemented fit method!")

    def predict(self, x_test):
        """
        Predicts the output for the given test data.

        Args:
            x_test (DataFrame): The input data for testing, processed in the same way that the fit method takes in.

        Returns:
            list of lists: Predicted trajectories, where each sublist corresponds to a predicted trajectory. The length
            of the sublist should be self.prediction_horizon // 5.
        """
        if not self.is_fitted:
            raise NotFittedError("This model instance is not fitted yet. Call 'fit' with appropriate arguments before "
                                 "using this estimator.".format(type(self).__name__))
        return self._predict_model(x_test)

    @abstractmethod
    def _predict_model(self, x_test):
        raise NotImplementedError("Model has not implemented predict method!")

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        raise NotImplementedError("Model has not implemented best_params method!")

    def process_data(self, df, model_config_manager: ModelConfigurationManager, real_time: bool):
        # Implement library specific preprocessing steps that are required before training a pandas dataframe
        raise NotImplementedError("Model has not implemented process_data method!")
