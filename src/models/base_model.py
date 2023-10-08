from sklearn.base import BaseEstimator, TransformerMixin


class BaseModel(BaseEstimator, TransformerMixin):
    def __init__(self, prediction_horizon):
        # TODO: Do we need this?
        # self.prediction_horizon = prediction_horizon
        pass

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
